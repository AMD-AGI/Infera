###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from typing import Any

import httpx
from sglang.srt.server_args import ServerArgs

from infera.common.net import free_tcp_port, free_tcp_port_block
from infera.common.worker_pool import DisaggMode, EngineType
from infera.engine.base import BaseEngine, EngineConfig

logger = logging.getLogger(__name__)


# SGLang uses "null" for mixed mode; we translate to our enum.
_SGLANG_TO_DISAGG_MODE = {
    "null": DisaggMode.MIXED,
    None: DisaggMode.MIXED,
    "prefill": DisaggMode.PREFILL,
    "decode": DisaggMode.DECODE,
}


class SglangEngine(BaseEngine):
    """Runs `python -m sglang.launch_server` in a child process.

    Earlier versions ran ``launch_server`` directly inside a daemon thread of
    the asyncio process. That breaks for any flag combination that triggers
    uvicorn's multi-worker supervisor (e.g. ``--tokenizer-worker-num 32``),
    because :func:`uvicorn.supervisors.multiprocess.Multiprocess.__init__`
    installs signal handlers via :func:`signal.signal`, which only works from
    the main thread of the main interpreter.

    Running sglang as a subprocess sidesteps that entirely and also gives us
    a clean ``SIGTERM`` / ``killpg`` shutdown story.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        sglang_argv: list[str] | None = None,
        advertise_host: str | None = None,
        *,
        enable_kv_events: bool = False,
    ) -> None:
        self.server_args = server_args
        # The exact argv we'll forward to `sglang.launch_server`. Forwarding
        # the original argv verbatim (instead of round-tripping through
        # ServerArgs) preserves multi-value flags such as
        # ``--cuda-graph-bs 1 2 3 ...``.
        self.sglang_argv = list(sglang_argv) if sglang_argv else []
        # IP/hostname peers should use to reach this worker. When sglang binds
        # on 0.0.0.0 (typical for multi-node deployments) we must publish a
        # routable address to etcd instead of 0.0.0.0.
        self.advertise_host = advertise_host or server_args.host
        self.enable_kv_events = enable_kv_events
        self._kv_events_port: int | None = None
        self._proc: subprocess.Popen | None = None

    async def start(self) -> EngineConfig:
        argv = list(self.sglang_argv)

        if self.enable_kv_events:
            dp_size = int(getattr(self.server_args, "dp_size", 1) or 1)
            self._kv_events_port = free_tcp_port_block(dp_size) if dp_size > 1 else free_tcp_port()
            kv_cfg = json.dumps(
                {
                    "publisher": "zmq",
                    "endpoint": f"tcp://*:{self._kv_events_port}",
                    "topic": "kv-events",
                }
            )
            argv += ["--kv-events-config", kv_cfg]

        cmd = [sys.executable, "-m", "sglang.launch_server", *argv]
        logger.info("spawning sglang subprocess: %s", " ".join(cmd))
        # start_new_session=True puts the child in its own process group so we
        # can SIGTERM the whole group (sglang spawns many helper processes).
        self._proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            start_new_session=True,
            stdout=None,
            stderr=None,
        )

        await self._wait_ready()

        disagg_mode = _SGLANG_TO_DISAGG_MODE[self.server_args.disaggregation_mode]
        disagg_meta: dict[str, Any] = {}
        if disagg_mode != DisaggMode.MIXED:
            # Both PREFILL and DECODE tag the protocol so the router can
            # fail-fast on accidental cross-protocol pairing (e.g.
            # SGLang prefill + vLLM-mooncake decode). Only PREFILL carries
            # the bootstrap endpoint; DECODE has nothing to advertise.
            params: dict[str, Any] = {}
            if disagg_mode == DisaggMode.PREFILL:
                params["bootstrap_addr"] = (
                    f"{self.advertise_host}:{self.server_args.disaggregation_bootstrap_port}"
                )
            disagg_meta = {"protocol": "sglang-bootstrap", "params": params}

        kv_events_endpoint: str | None = None
        kv_block_size: int | None = None
        if self.enable_kv_events:
            # Advertise the ZMQ endpoint on the routable address, not 0.0.0.0.
            kv_events_endpoint = f"tcp://{self.advertise_host}:{self._kv_events_port}"
            kv_block_size = self.server_args.page_size

        # Native DP: one endpoint fronts dp_size internal ranks. Register the
        # size (rank-multiplexed; dp_rank stays None) so the router can expand
        # into per-rank targets and steer with X-Data-Parallel-Rank.
        dp_size = self.server_args.dp_size if (self.server_args.dp_size or 1) > 1 else None

        return EngineConfig(
            model_name=self.server_args.served_model_name or self.server_args.model_path,
            host=self.advertise_host,
            port=self.server_args.port,
            engine=EngineType.SGLANG,
            disagg_mode=disagg_mode,
            disagg_meta=disagg_meta,
            kv_events_endpoint=kv_events_endpoint,
            kv_block_size=kv_block_size,
            dp_size=dp_size,
        )

    async def _wait_ready(self, timeout: float = 1800) -> None:
        # /health is probed locally; sglang binds on server_args.host, but if
        # that is 0.0.0.0 we should probe via 127.0.0.1 instead.
        probe_host = self.server_args.host
        if probe_host in ("0.0.0.0", ""):
            probe_host = "127.0.0.1"
        url = f"http://{probe_host}:{self.server_args.port}/health"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_log = 0.0
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"sglang subprocess exited with code {self._proc.returncode} "
                        "before reporting ready"
                    )
                try:
                    r = await client.get(url, timeout=5)
                    if r.status_code == 200:
                        logger.info("SGLang ready on port %d", self.server_args.port)
                        return
                except httpx.HTTPError:
                    # Covers ConnectError (server not up yet) and ReadTimeout
                    # (server up but busy compiling AITER kernels). Both are
                    # normal during startup; keep polling.
                    pass
                now = loop.time()
                if now - last_log >= 30.0:
                    logger.info(
                        "waiting for SGLang HTTP on port %d ... (elapsed %.0fs)",
                        self.server_args.port,
                        now - (deadline - timeout),
                    )
                    last_log = now
                await asyncio.sleep(2)
        raise TimeoutError(f"SGLang not ready after {timeout}s")

    async def stop(self) -> None:
        logger.info("SGLang engine stopping")
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        # Give the child up to 30s for a graceful exit, then escalate.
        for _ in range(30):
            if self._proc.poll() is not None:
                return
            await asyncio.sleep(1)
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
