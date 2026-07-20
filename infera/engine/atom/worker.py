###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""AtomEngine — runs ATOM's OpenAI server as a subprocess.

Mirrors :class:`infera.engine.vllm.worker.VllmEngine` /
:class:`infera.engine.sglang.worker.SglangEngine`: spawn the engine in
its own process group, poll ``/health`` until ready, and report an
:class:`EngineConfig` back to the launcher for etcd registration.

KV-aware routing is **off by default**. ATOM has no native KV-event stream,
so when the operator opts in with ``--enable-kv-events`` infera installs a
BlockManager hook (see :mod:`infera.engine.atom.hooks.kv_events`) inside the
ATOM subprocess that republishes ATOM's prefix-cache index on a ZMQ PUB
socket in the router's wire format: the launcher passes the bind endpoint to
the subprocess via ``INFERA_ATOM_KV_EVENTS_ENDPOINT`` and advertises the
reachable endpoint + block size in :class:`EngineConfig`. When not enabled
(the default) the config carries no ``kv_events_endpoint`` and the router
routes the worker round-robin.
"""

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


class AtomEngine(BaseEngine):
    def __init__(
        self,
        *,
        atom_argv: list[str],
        model_name: str,
        host: str,
        port: int,
        advertise_host: str | None = None,
        kv_events_endpoint: str | None = None,
        kv_events_bind: str | None = None,
        kv_block_size: int | None = None,
        disagg_mode: DisaggMode = DisaggMode.MIXED,
        disagg_meta: dict[str, Any] | None = None,
    ) -> None:
        self.atom_argv = list(atom_argv)
        self.model_name = model_name
        self.host = host
        self.port = port
        self.advertise_host = advertise_host or host
        # ``kv_events_endpoint`` is what the router connects to (advertised);
        # ``kv_events_bind`` is what the ATOM subprocess binds (``tcp://*:port``).
        self.kv_events_endpoint = kv_events_endpoint
        self.kv_events_bind = kv_events_bind
        self.kv_block_size = kv_block_size
        self.disagg_mode = disagg_mode
        self.disagg_meta: dict[str, Any] = dict(disagg_meta or {})
        self._proc: subprocess.Popen | None = None

    async def start(self) -> EngineConfig:
        cmd = [
            sys.executable,
            "-m",
            "atom.entrypoints.openai_server",
            *self.atom_argv,
        ]
        logger.info("spawning atom subprocess: %s", " ".join(cmd))
        env = os.environ.copy()
        # Hand the ZMQ bind endpoint to the ATOM subprocess. The site-startup
        # hook (infera.engine.atom.hooks.kv_event_bootstrap, run via a .pth in
        # every interpreter — including ATOM's spawned EngineCore) reads this
        # and installs the BlockManager KV-event publisher.
        if self.kv_events_bind:
            env["INFERA_ATOM_KV_EVENTS_ENDPOINT"] = self.kv_events_bind
            logger.info(
                "ATOM kv-events enabled: bind=%s advertise=%s block_size=%s",
                self.kv_events_bind,
                self.kv_events_endpoint,
                self.kv_block_size,
            )
        # start_new_session=True puts the child in its own process group so we
        # can SIGTERM the whole group (ATOM spawns TP worker processes).
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
            stdout=None,
            stderr=None,
        )

        await self._wait_ready()

        return EngineConfig(
            model_name=self.model_name,
            host=self.advertise_host,
            port=self.port,
            engine=EngineType.ATOM,
            disagg_mode=self.disagg_mode,
            disagg_meta=dict(self.disagg_meta),
            kv_events_endpoint=self.kv_events_endpoint,
            kv_block_size=self.kv_block_size,
        )

    async def stop(self) -> None:
        logger.info("ATOM engine stopping")
        if self._proc is None:
            return
        # The subprocess is a session leader (start_new_session=True), so its
        # pid doubles as the process-group id for the whole ATOM tree.
        pgid = self._proc.pid
        if self._proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            # Graceful window kept short so we always reach the SIGKILL sweep
            # within the launcher's own teardown budget (the parent harness
            # SIGKILLs this process group ~25s after SIGTERM).
            for _ in range(15):
                if self._proc.poll() is not None:
                    break
                await asyncio.sleep(1)
        # The leader (ATOM's openai_server) exiting does NOT mean the whole
        # group is gone: ATOM/AITER helpers — e.g. the shared-memory broadcast
        # worker — can hang and ignore SIGTERM, holding GPU VRAM until the
        # container dies and starving the next worker (HIP-OOM). Always sweep
        # the group with SIGKILL so VRAM is released on teardown.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            pass

    async def _wait_ready(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = float(os.environ.get("INFERA_ATOM_READY_TIMEOUT", "1800"))
        probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        url = f"http://{probe_host}:{self.port}/health"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_log = 0.0
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"atom subprocess exited with code {self._proc.returncode} "
                        "before reporting ready"
                    )
                try:
                    r = await client.get(url, timeout=5)
                    if r.status_code == 200:
                        logger.info("ATOM ready on port %d", self.port)
                        return
                except httpx.HTTPError:
                    pass
                now = loop.time()
                if now - last_log >= 30.0:
                    logger.info(
                        "waiting for ATOM HTTP on port %d ... (elapsed %.0fs)",
                        self.port,
                        now - (deadline - timeout),
                    )
                    last_log = now
                await asyncio.sleep(2)
        raise TimeoutError(f"ATOM not ready after {timeout}s")
