###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from infera.common.worker_pool import DisaggMode, EngineType, KvRegistrationMetadata

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Returned by engine after startup, describes its capabilities."""

    model_name: str
    host: str
    port: int
    engine: EngineType = EngineType.SGLANG
    disagg_mode: DisaggMode = DisaggMode.MIXED
    disagg_meta: dict[str, Any] = field(default_factory=dict)
    # KV management block — populated by engines that have wired in a
    # KvEventProbe. Leave None for engines that don't yet participate
    # in the KV index; they'll still register and route, just without
    # prefix-cache scoring.
    kv: KvRegistrationMetadata | None = None

    # KV-aware routing
    kv_events_endpoint: str | None = None
    kv_block_size: int | None = None

    # Data parallel
    dp_rank: int | None = None
    dp_size: int | None = None

    # Request transport the router should use to reach this worker
    # ("http" direct forward, or "nats" per-instance subject). Set by the
    # worker entrypoint based on --request-transport.
    request_transport: str = "http"


class BaseEngine(ABC):
    # Every engine runs its server as a child process (subprocess.Popen) and
    # stores it here, so the base class can offer process-lifecycle helpers
    # (e.g. wait()) uniformly across sglang / vllm / atom.
    _proc: subprocess.Popen | None = None

    @abstractmethod
    async def start(self) -> EngineConfig:
        """Start the inference engine, return its config once ready."""
        ...

    @abstractmethod
    async def stop(self) -> None: ...

    async def wait(self, poll_interval: float = 1.0) -> int | None:
        proc = self._proc
        if proc is None:
            return None
        while proc.poll() is None:
            await asyncio.sleep(poll_interval)
        return proc.returncode


def watch_engine_death(engine: BaseEngine, stop: asyncio.Event) -> asyncio.Task:
    async def _watch() -> None:
        if engine._proc is None:
            # No subprocess to watch (e.g. engine not started yet); never trip
            # shutdown on a phantom "death".
            return
        code = await engine.wait()
        logger.error(
            "engine subprocess exited (code=%s); deregistering worker and shutting down",
            code,
        )
        stop.set()

    return asyncio.create_task(_watch(), name="engine-death-watch")
