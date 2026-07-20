###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared pytest fixtures for the per-engine e2e suites.

``make_worker_fixture(adapter_factory)`` builds the ``worker`` factory fixture
each engine's conftest binds to its own :class:`~.adapter.EngineAdapter`, so
the spawn / readiness-poll / teardown / GPU-allocation logic lives here once
instead of being copy-pasted per engine.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest_asyncio

from .adapter import EngineAdapter, GpuAllocator, spawn_worker, teardown_workers
from .params import EngineParams
from .resources import visible_gpu_count


def make_worker_fixture(adapter_factory: Callable[[], EngineAdapter]):
    """Return a function-scoped ``worker`` fixture bound to ``adapter_factory``.

    The fixture yields an async factory ``await worker(server, params)`` that
    returns a :class:`~.adapter.WorkerHandle`. GPUs are allocated disjointly
    across all workers spawned within a single test; all are torn down after.
    Each spawn first waits for its GPUs' VRAM to be released (see spawn_worker),
    so back-to-back cases reusing the same GPUs don't hit a spurious HIP-OOM.
    """

    @pytest_asyncio.fixture
    async def worker():
        adapter = adapter_factory()
        alloc = GpuAllocator(total=max(1, visible_gpu_count()))
        procs: list = []

        async def _spawn(server_ctx: dict, params: EngineParams | None = None):
            params = params or EngineParams()
            gpu_ids = alloc.take(adapter.gpus_per_worker(params))
            return await spawn_worker(adapter, server_ctx, params, gpu_ids=gpu_ids, procs=procs)

        yield _spawn

        teardown_workers(procs)

    return worker
