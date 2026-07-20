###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared "disaggregated" (prefill/decode split across nodes) e2e building block.

:func:`run_disagg_case` is the engine-agnostic body of the parametrized
``test_disagg`` (guards + bring up the full containerized stack + the same
correctness probes as PD-mixed). Each engine's ``test_disagg.py`` wraps it with
its own parametrize list; the per-engine ``disagg_stack`` fixture (bound via
:func:`tests.e2e.harness.disagg_fixtures.make_disagg_stack_fixture`) owns the
node placement + launch of etcd/router/prefill/decode, and the engine adapter
owns the role-aware argv/env.

Placement/topology (SLURM) and the RDMA KV transport live in :mod:`.cluster` /
:mod:`.launcher`; correctness reuses :mod:`.scenarios` unchanged (the assertions
only talk to the router's public HTTP surface, so P/D routing is transparent).
Nothing infera runs on the driver host — it only drives srun/docker + probes.
"""

from __future__ import annotations

from . import resources, scenarios
from .params import EngineParams

__all__ = ["run_disagg_case"]


async def run_disagg_case(params: EngineParams, disagg_stack) -> None:
    """Shared body: skip unsupported combos / environments, bring up the whole
    containerized stack (etcd + router + prefill on node 0, decode on node 1),
    then verify chat liveness + semantic correctness end-to-end (the request is
    routed P->D with the KV cache transferred over RDMA)."""
    resources.require_supported(params)

    # Brings up the full stack across two nodes and returns the router context;
    # self-skips if the disagg environment (SLURM/allocation/nodes) is missing
    # or the engine has no PD adapter yet.
    server = await disagg_stack(params)

    # Correctness only (no standalone chat liveness): assert_correctness passes on
    # counting (/v1/completions) OR capital (/v1/chat/completions), and its capital
    # probe is tolerant — so completions-only PD engines (e.g. ATOM) pass on the
    # counting probe alone while the P->D KV transfer is still exercised end-to-end.
    await scenarios.assert_correctness(server["url"], params.model)
