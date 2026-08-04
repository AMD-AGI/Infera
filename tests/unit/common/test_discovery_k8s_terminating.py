###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A condemned Pod must leave the pool before its process is killed.

Kubernetes keeps ``phase: Running`` on a terminating Pod until its containers
exit, so liveness alone cannot tell a healthy worker from one that is seconds
from SIGTERM. The operator makes that window long on purpose -- it injects a
``preStop sleep`` so in-flight work has time to finish -- which means that
without a ``deletionTimestamp`` check the router spends the entire drain window
assigning new requests to a worker that is guaranteed to be killed.

These tests pin the removal rules rather than the implementation: what matters
is which observable Pod states take a worker out of rotation.
"""

from __future__ import annotations

import json

from infera.common.discovery_k8s import WORKER_INFO_ANNOTATION, KubernetesRegistry


def _payload(worker_id: str = "10.0.0.1:8080") -> str:
    host, port = worker_id.split(":")
    return json.dumps(
        {
            "worker_id": worker_id,
            "url": f"http://{worker_id}",
            "model_name": "m",
            "engine": "sglang",
            "disagg_mode": "mixed",
            "disagg_meta": {},
            "kv_events_endpoint": None,
            "kv_block_size": None,
            "dp_rank": None,
            "dp_size": None,
            "request_transport": "http",
        }
    )


def _pod(name="w-0", *, annotated=True, phase="Running", terminating=False):
    meta: dict = {"name": name}
    if annotated:
        meta["annotations"] = {WORKER_INFO_ANNOTATION: _payload()}
    if terminating:
        meta["deletionTimestamp"] = "2026-08-04T03:00:00Z"
    return {"metadata": meta, "status": {"phase": phase}}


def _registry():
    removed: list[str] = []
    reg = KubernetesRegistry(
        "app=infera",
        namespace="infera",
        on_worker_removed=removed.append,
    )
    return reg, removed


def _ids(reg):
    return [w.worker_id for w in reg.pool.list_active()]


def test_running_pod_registers():
    reg, _ = _registry()
    reg._handle_pod(_pod(), deleted=False)
    assert _ids(reg) == ["10.0.0.1:8080"]


def test_terminating_pod_is_removed_while_still_running():
    """The case this exists for. `phase` is still Running -- only the deletion
    timestamp distinguishes a healthy worker from one inside its preStop delay,
    and every request routed to it in that window is work that gets cut."""
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    assert _ids(reg) == ["10.0.0.1:8080"]

    reg._handle_pod(_pod(terminating=True, phase="Running"), deleted=False)
    assert _ids(reg) == [], "a condemned Pod must not stay a routing candidate"
    assert removed == ["10.0.0.1:8080"]


def test_terminating_pod_never_enters_the_pool():
    """A relist during a rolling update can surface an already-terminating Pod
    the registry has never seen. It must not be admitted."""
    reg, _ = _registry()
    reg._handle_pod(_pod(terminating=True), deleted=False)
    assert _ids(reg) == []


def test_removal_is_idempotent():
    """Watch events are re-delivered after a 410/relist, so the same
    terminating Pod arrives more than once."""
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    for _ in range(3):
        reg._handle_pod(_pod(terminating=True), deleted=False)
    assert removed == ["10.0.0.1:8080"], "must not fire the removal callback repeatedly"


def test_other_removal_rules_still_hold():
    for label, kwargs, deleted in (
        ("explicit DELETE", {}, True),
        ("annotation cleared", {"annotated": False}, False),
        ("no longer Running", {"phase": "Failed"}, False),
    ):
        reg, _ = _registry()
        reg._handle_pod(_pod(), deleted=False)
        reg._handle_pod(_pod(**kwargs), deleted=deleted)
        assert _ids(reg) == [], f"{label} must still deregister"
