###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""VllmEngine — runs `vllm serve` as a subprocess. Mirrors SglangEngine."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from typing import Any

import httpx

from infera.common.worker_pool import DisaggMode, EngineType
from infera.engine.base import BaseEngine, EngineConfig

logger = logging.getLogger(__name__)


class VllmEngine(BaseEngine):
    def __init__(
        self,
        *,
        vllm_argv: list[str],
        model_name: str,
        host: str,
        port: int,
        advertise_host: str | None = None,
        kv_events_endpoint: str | None = None,
        kv_block_size: int | None = None,
        disagg_mode: DisaggMode = DisaggMode.MIXED,
        disagg_meta: dict[str, Any] | None = None,
        dp_rank: int | None = None,
        dp_size: int | None = None,
    ) -> None:
        self.vllm_argv = list(vllm_argv)
        self.model_name = model_name
        self.host = host
        self.port = port
        self.advertise_host = advertise_host or host
        self.kv_events_endpoint = kv_events_endpoint
        self.kv_block_size = kv_block_size
        self.disagg_mode = disagg_mode
        # ``{}`` for MIXED workers and connectors without a registered
        # DisaggProtocol; ``{"protocol": ..., "params": ...}`` otherwise.
        # Computed once by the launcher (infera.engine.vllm.args) from
        # the parsed --kv-transfer-config.
        self.disagg_meta: dict[str, Any] = dict(disagg_meta or {})
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self._proc: subprocess.Popen | None = None

    async def start(self) -> EngineConfig:
        # Equivalent to `vllm serve` but doesn't depend on the console script on PATH.
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            *self.vllm_argv,
        ]
        logger.info("spawning vllm subprocess: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            start_new_session=True,
            stdout=None,
            stderr=None,
        )

        await self._wait_ready()

        return EngineConfig(
            model_name=self.model_name,
            host=self.advertise_host,
            port=self.port,
            engine=EngineType.VLLM,
            disagg_mode=self.disagg_mode,
            disagg_meta=dict(self.disagg_meta),
            kv_events_endpoint=self.kv_events_endpoint,
            kv_block_size=self.kv_block_size,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
        )

    async def stop(self) -> None:
        logger.info("vLLM engine stopping")
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(30):
            if self._proc.poll() is not None:
                return
            await asyncio.sleep(1)
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _wait_ready(self, timeout: float = 1800) -> None:
        # 30 min: cold model download + ROCm kernel compile can eat 10+ min.
        probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        url = f"http://{probe_host}:{self.port}/health"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_log = 0.0
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"vllm subprocess exited with code {self._proc.returncode} "
                        "before reporting ready"
                    )
                try:
                    r = await client.get(url, timeout=5)
                    if r.status_code == 200:
                        logger.info("vLLM ready on port %d", self.port)
                        return
                except httpx.HTTPError:
                    pass
                now = loop.time()
                if now - last_log >= 30.0:
                    logger.info(
                        "waiting for vLLM HTTP on port %d ... (elapsed %.0fs)",
                        self.port,
                        now - (deadline - timeout),
                    )
                    last_log = now
                await asyncio.sleep(2)
        raise TimeoutError(f"vLLM not ready after {timeout}s")
