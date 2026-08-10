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
from infera.common.worker_pool import WorkerStatus


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


def _all(reg):
    return {w.worker_id: w.status for w in reg.pool.list_all()}


def test_a_draining_worker_stays_visible():
    """Out of routing, still on the books.

    Dropping the record entirely makes a worker finishing its in-flight
    generations look exactly like one that crashed, so `/v1/workers` cannot
    tell an orderly rollout from a fleet losing workers -- at exactly the
    moment someone is watching one happen.
    """
    reg, _ = _registry()
    reg._handle_pod(_pod(), deleted=False)
    reg._handle_pod(_pod(terminating=True), deleted=False)

    assert _ids(reg) == [], "still must not be a routing candidate"
    assert _all(reg) == {"10.0.0.1:8080": WorkerStatus.DRAINING}


def test_the_record_goes_when_the_worker_deregisters():
    """The worker clears its own annotation on SIGTERM, before draining, which
    lands here as 'annotation gone'. Without that the draining record would be
    immortal."""
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    reg._handle_pod(_pod(terminating=True), deleted=False)
    assert _all(reg), "precondition: the record survives the terminating event"

    reg._handle_pod(_pod(terminating=True, annotated=False), deleted=False)
    assert _all(reg) == {}, "a drained worker must leave the pool"
    assert removed == ["10.0.0.1:8080"], "announced once, not once per stage"


def test_a_worker_killed_without_its_pod_being_deleted_is_announced_once():
    """The path deregistering-before-draining exists for.

    A liveness probe restarting the container, a node shutting down gracefully,
    someone killing the process -- all send SIGTERM with the Pod object
    untouched. There is no deletionTimestamp, so the annotation the worker
    clears on its way out is the only signal, and the callback behind it is what
    stops the KV subscriber. A later DELETE for the same Pod must not repeat it.
    """
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)

    reg._handle_pod(_pod(annotated=False), deleted=False)
    assert _ids(reg) == [], "clearing the annotation must stop new work arriving"
    assert removed == ["10.0.0.1:8080"], "the only signal on this path must announce"

    reg._handle_pod(_pod(annotated=False), deleted=True)
    assert removed == ["10.0.0.1:8080"], "the eventual DELETE must not announce again"


def test_announced_once_across_draining_then_delete():
    """The callbacks stop a KV subscriber and clear block accounting. Firing
    them twice for one worker is not free, and firing them late (only at the
    DELETE) would leave the router accounting for a worker it no longer routes
    to for the whole drain."""
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    reg._handle_pod(_pod(terminating=True), deleted=False)
    assert removed == ["10.0.0.1:8080"], "must announce as soon as it leaves routing"

    reg._handle_pod(_pod(terminating=True), deleted=True)
    assert _all(reg) == {}
    assert removed == ["10.0.0.1:8080"], "the DELETE must not re-announce"


def test_a_worker_that_never_drained_still_announces_on_delete():
    """The drain path is not the only way out: a crash or an evicted Pod goes
    straight to removal, and that still has to reach the callbacks."""
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    reg._handle_pod(_pod(), deleted=True)
    assert _all(reg) == {}
    assert removed == ["10.0.0.1:8080"]


def test_a_list_reconciles_away_a_worker_whose_delete_was_missed():
    """A list is a complete snapshot, so a tracked Pod it does not mention is
    gone.

    This is not hypothetical. Keeping a draining record alive means its removal
    now depends on a later event, and the watch drops out routinely -- the
    re-list exists precisely because etcd compaction expires the
    resourceVersion every few minutes. A Pod deleted inside that window
    produces no event anyone sees, so without reconciling against the list the
    record is immortal: `/v1/workers` reports a worker that does not exist, and
    the model's canary is never forgotten because a phantom still holds it.
    """
    reg, removed = _registry()
    reg._handle_pod(_pod(), deleted=False)
    reg._handle_pod(_pod(terminating=True), deleted=False)
    assert _all(reg), "precondition: the draining record is being kept"

    # The Pod is gone; a fresh list simply does not contain it.
    reg._reconcile_absent(seen=set())

    assert _all(reg) == {}, "a tracked Pod missing from a full list must be dropped"
    assert removed == ["10.0.0.1:8080"], "already announced when it started draining"


def test_a_list_keeps_workers_it_still_sees():
    reg, _ = _registry()
    reg._handle_pod(_pod(name="w-0"), deleted=False)
    reg._reconcile_absent(seen={"w-0"})
    assert _ids(reg) == ["10.0.0.1:8080"], "a Pod present in the list must survive"
