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


def _ready_timeout() -> float:
    """Seconds to wait for the engine's /health, from INFERA_ENGINE_READY_TIMEOUT."""
    try:
        return float(os.environ.get("INFERA_ENGINE_READY_TIMEOUT", "1800"))
    except ValueError:
        return 1800.0


# The resolved page size is read once per process and nothing re-resolves it,
# so a single transient failure disables kv-aware for that pod permanently.
# Short and bounded: the engine is already serving by the time this runs, so
# anything needing more than a few seconds is not transient.
_PAGE_SIZE_ATTEMPTS = 4
_PAGE_SIZE_BACKOFF_S = 1.0


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
        # Filled from /get_server_info once the subprocess is up; see
        # `_resolve_page_size`.
        self.resolved_page_size: int | None = None

    async def start(self) -> EngineConfig:
        argv = list(self.sglang_argv)

        # sglang serves /metrics only with --enable-metrics; without it the
        # endpoint 404s. Graceful shutdown reads the in-flight request count
        # from there, so leaving it off silently downgrades every scale-down
        # and rolling update to "kill in-flight generations". Cheap enough to
        # always enable, and the caller can still have passed it explicitly.
        if not any(a == "--enable-metrics" for a in argv):
            argv.append("--enable-metrics")

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
        self.resolved_page_size = await self._resolve_page_size()

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
            # Resolved value first: `server_args.page_size` is what the operator
            # typed, and it is None whenever they left it to the engine. The
            # router builds this worker's KV view at whatever number we register
            # here, then rejects every event that disagrees -- so a guess is not
            # a safe default, it is a silent 0% hit rate. See `_resolve_page_size`.
            kv_block_size = self.resolved_page_size or self.server_args.page_size

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

    # Weight-load time tracks the storage, not the model: Kimi-K3 read 96 shards in
    # 502 s from local NVMe and ~95 min from NFS with both PD nodes competing for
    # the same mount. A hardcoded 1800 s is generous for the first and impossible
    # for the second — the worker killed itself mid-load, restarted, and could
    # never finish. Tunable via INFERA_ENGINE_READY_TIMEOUT (seconds).
    async def _wait_ready(self, timeout: float = _ready_timeout()) -> None:
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

    async def _resolve_page_size(self) -> int | None:
        """The KV page size the engine actually settled on, or None.

        `server_args.page_size` is only the operator's request. Several of
        sglang's post-process passes run inside the launch_server subprocess --
        the DSA one needs a ModelConfig -- so when the flag is left off, our copy
        stays None while the engine lands on something else entirely. Seen on
        GLM-5.3-Flash (attention_backend=dsa): we registered a fabricated 1, the
        engine paged at 64, and the router silently dropped to load-only routing
        for both legs with every health signal green.

        Guessing cannot fix that; asking can. The subprocess is already serving
        by the time this runs (`_wait_ready` returned), and /get_server_info
        reports the resolved ServerArgs. Best-effort on purpose: an older sglang
        without the route, or a malformed body, leaves the caller on the flag --
        no worse than before, and the caller registers None rather than a lie.

        Retried, because a single miss is permanent. This runs once per process
        and nothing re-resolves afterwards, so when `--page-size` is also unset
        one slow or 500ing moment -- an engine that is /health-ready but still
        warming under load -- registers `engine_block_size=None` AND refuses to
        start the KV NATS relay, for the entire lifetime of that pod. A rolling
        restart could strip kv-aware from an arbitrary fraction of the fleet
        that way. `_wait_ready` directly above already polls for the same
        reason; this had the same need and not the loop.
        """
        probe_host = self.server_args.host
        if probe_host in ("0.0.0.0", "", "::"):
            probe_host = "127.0.0.1"
        url = f"http://{probe_host}:{self.server_args.port}/get_server_info"
        info = None
        last_exc: Exception | None = None
        for attempt in range(_PAGE_SIZE_ATTEMPTS):
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, timeout=10)
                    r.raise_for_status()
                    info = r.json()
                break
            except Exception as exc:  # noqa: BLE001 - retried, then reported below
                last_exc = exc
                if attempt + 1 < _PAGE_SIZE_ATTEMPTS:
                    logger.debug(
                        "page_size probe %d/%d failed (%s); retrying",
                        attempt + 1,
                        _PAGE_SIZE_ATTEMPTS,
                        exc,
                    )
                    await asyncio.sleep(_PAGE_SIZE_BACKOFF_S * (attempt + 1))
        if info is None:
            exc = last_exc
            logger.warning(
                "could not read resolved page_size from %s (%s); falling back to "
                "--page-size (%r). If that is None this worker registers no KV block "
                "size and kv-aware routing skips it -- pass --page-size to pin it.",
                url,
                exc,
                self.server_args.page_size,
            )
            return None
        # The route has moved between sglang versions: some builds return the
        # ServerArgs fields at the top level, others nest them under a key.
        for scope in (info, info.get("server_args") if isinstance(info, dict) else None):
            if isinstance(scope, dict) and scope.get("page_size"):
                resolved = int(scope["page_size"])
                requested = self.server_args.page_size
                if requested and int(requested) != resolved:
                    logger.warning(
                        "sglang resolved page_size=%d, overriding the requested %s; "
                        "registering the resolved value so the router's KV view matches "
                        "the events this engine emits",
                        resolved,
                        requested,
                    )
                else:
                    logger.info("sglang resolved page_size=%d", resolved)
                return resolved
        logger.warning(
            "%s carried no page_size; falling back to --page-size (%r)",
            url,
            self.server_args.page_size,
        )
        return None

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
