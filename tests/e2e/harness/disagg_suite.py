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

import re

from . import resources, scenarios, speculation
from .adapter import emit_reporter_line
from .params import EngineParams

__all__ = ["run_disagg_case"]

# The two ways Mooncake reports its transport. Either fallback still delivers the
# KV over sockets and still passes the correctness probes, so the log is the only
# place a run that never touched RDMA can be told apart from one that did.
_TRANSPORT_BANNER = re.compile(r"installTransport, type=(\w+)")
_NO_HCA = re.compile(r"Topology discovery complete\. Found 0 HCAs")


def assert_rdma_kv_transport(server: dict) -> None:
    """Fail if a PD worker moved KV over anything but RDMA. A worker whose log
    says nothing about Mooncake is reported "unverified" rather than failed, so
    this never blocks an engine whose transport it cannot read."""
    launcher, workers = server.get("launcher"), server.get("workers") or []
    for h in workers:
        log = launcher.collect_logs(h)
        where = f"{h.role} @ {h.node}"
        assert not _NO_HCA.search(log), (
            f"{where}: Mooncake found 0 RDMA devices and served KV over TCP — the "
            f"engine asked for a NIC this fabric does not have; see {h.log_path}"
        )
        kinds = sorted(set(_TRANSPORT_BANNER.findall(log)))
        if not kinds:
            emit_reporter_line(f"[e2e disagg transport] {where}: unverified (no Mooncake log)")
            continue
        emit_reporter_line(f"[e2e disagg transport] {where}: {', '.join(kinds)}")
        assert kinds == ["rdma"], (
            f"{where} did not run KV over RDMA alone (installed: {', '.join(kinds)}); "
            f"see {h.log_path}"
        )


async def run_disagg_case(params: EngineParams, disagg_stack) -> None:
    """Shared body: skip unsupported combos / environments, bring up the whole
    containerized stack (etcd + router + prefill on node 0, decode on node 1),
    then verify chat liveness + semantic correctness end-to-end (the request is
    routed P->D with the KV cache transferred over RDMA)."""
    resources.require_supported(params)
    # Before the image build, not after: on a fleet where the model tree is shared
    # NFS the orchestrator can read it too, so a staging mistake is catchable here
    # rather than 40 minutes later on both nodes at once.
    resources.require_model_staged(params)

    # Brings up the full stack across two nodes and returns the router context;
    # self-skips if the disagg environment (SLURM/allocation/nodes) is missing
    # or the engine has no PD adapter yet.
    server = await disagg_stack(params)

    # Correctness only (no standalone chat liveness). Its chat-based probes self-report
    # as not-run on completions-only PD engines (e.g. ATOM), leaving counting and the
    # long-context retrieval — the latter being what makes the P->D hop move real KV.
    await scenarios.assert_correctness(server["url"], params.model)

    # …and that it got there over RDMA, which correctness alone cannot show.
    assert_rdma_kv_transport(server)

    # Speculation, read off the decode leg: that is the only one that drafts.
    # The prefill leg runs a single forward per request and would report zero
    # even on a perfectly healthy MTP setup, so asserting there would be wrong.
    decode = next(
        (w for w in server.get("workers", ()) if getattr(w, "role", "") == "decode"),
        None,
    )
    if decode is not None:
        await speculation.report_speculation(
            decode.port,
            params,
            engine=server.get("engine", "?"),
            host=decode.advertise_host,
        )
