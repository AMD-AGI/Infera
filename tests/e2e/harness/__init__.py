###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine-agnostic e2e harness (shared by the PD-mixed + PD-disaggregated suites).

Public surface used by the per-engine test suites:

    EngineParams        the high-level knob set under test
    DisaggRole          prefill/decode role for PD-disaggregated workers
    EngineAdapter       per-engine argv/env translation contract (mixed + disagg)
    WorkerHandle        a spawned local worker's id/port/proc/gpu handle
    GpuAllocator        disjoint GPU-index allocation within a test
    spawn_worker        launch a LOCAL worker + wait until active (PD-mixed)
    teardown_workers    graceful SIGTERM/SIGKILL of spawned local workers
    running_server      in-process infera server (round-robin)
    client              OpenAI-compatible HTTP helpers
    scenarios           reusable flow + assertions (run_mixed, ...)
    resources           GPU/model guards (require_gpus, require_supported)

PD-disaggregated (cross-node) placement lives in :mod:`.cluster` (SLURM topology)
+ :mod:`.launcher` (``SrunDockerLauncher``); the shared bodies are
:mod:`.mixed_suite` (``run_mixed_case``) and :mod:`.disagg_suite`
(``run_disagg_case``). Shared parametrize building blocks live in :mod:`.matrix`
(a declarative ``[model, tp, ep, dp_attn]`` case table expanded by
``expand_cases``); each engine's own grid lives in its dir under
``tests/e2e/pd_mixed/`` / ``tests/e2e/pd_disag/``.
"""

from __future__ import annotations

from . import client, correctness, resources, scenarios
from .adapter import (
    EngineAdapter,
    GpuAllocator,
    WorkerHandle,
    spawn_worker,
    teardown_workers,
)
from .params import DEFAULT_MODEL, DisaggRole, EngineParams
from .resources import require_gpus, require_supported, visible_gpu_count

__all__ = [
    "EngineParams",
    "DisaggRole",
    "DEFAULT_MODEL",
    "EngineAdapter",
    "GpuAllocator",
    "WorkerHandle",
    "spawn_worker",
    "teardown_workers",
    "running_server",
    "client",
    "correctness",
    "scenarios",
    "resources",
    "require_gpus",
    "require_supported",
    "visible_gpu_count",
]


def __getattr__(name: str):
    # Lazy re-export of the in-process server (PD-mixed only). Importing it pulls
    # uvicorn + the infera.server stack, which the PD-disaggregated driver host
    # doesn't have (there every service runs in a container). Keeping it lazy
    # lets `tests/e2e/pd_disag` collect on a host with only pytest + httpx.
    if name == "running_server":
        from .server import running_server

        return running_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
