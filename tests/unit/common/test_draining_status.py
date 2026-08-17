###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Where DRAINING comes from, and why the registration record has no state.

``WorkerStatus.DRAINING`` takes a worker out of ``list_active`` while leaving it
visible, which is what distinguishes an orderly rollout from a crash. What sets
it is the Kubernetes registry, from the Pod's ``deletionTimestamp`` -- not the
worker.

That is a deliberate split. Kubernetes knows a Pod is condemned before the
process is signalled, so the orchestrator answers "is this worker leaving"
earlier and more authoritatively than the worker could. etcd has no such signal:
a record is present or absent, so there removing it is what stops new work, and
a shutdown deregisters before it drains. Either way the record itself carries
identity only, which is what makes the heartbeat -- which rebuilds it from
config -- safe to run at any point.
"""

from __future__ import annotations

import pytest

from infera.common.discovery import worker_info_from_json
from infera.common.registration import build_worker_payload
from infera.common.worker_pool import EngineType, WorkerPool, WorkerStatus
from infera.engine.base import EngineConfig


def _cfg():
    return EngineConfig(model_name="m", host="10.0.0.1", port=8080, engine=EngineType.SGLANG)


def test_the_record_carries_identity_and_no_state():
    """Every field is fixed for the life of the process, so two builds are
    byte-identical. State written here would be erased by the next heartbeat,
    which rebuilds the payload from the same config."""
    payload = build_worker_payload(_cfg())
    assert "status" not in payload
    assert payload == build_worker_payload(_cfg())


def test_a_record_without_a_status_reads_as_active():
    """Registration says nothing about state, so the parser's default is what
    every healthy worker resolves to."""
    info = worker_info_from_json(build_worker_payload(_cfg()))
    assert info.status is WorkerStatus.ACTIVE


def test_a_draining_worker_is_excluded_but_still_visible():
    """The value DRAINING adds over deleting the record is not routing -- both
    stop new work -- it is that the worker stays observable while it finishes."""
    pool = WorkerPool()
    pool.add(worker_info_from_json(build_worker_payload(_cfg())))
    assert [w.worker_id for w in pool.list_active()] == ["10.0.0.1:8080"]

    worker = pool.get("10.0.0.1:8080")
    worker.status = WorkerStatus.DRAINING
    assert pool.list_active() == [], "a draining worker must not be routed to"
    assert pool.get("10.0.0.1:8080") is not None, "but it must still be observable"


# --- which step stops new work arriving ---------------------------------------


def test_no_client_announces_a_status():
    """Neither backend writes state into the record any more.

    Under Kubernetes it was never read -- the registry acts on deletionTimestamp
    and returns before parsing the annotation -- while the heartbeat rebuilt the
    payload and erased it. On etcd it was read, but deregistering already stops
    new work, so announcing first only added a second mechanism for the same
    thing.
    """
    from infera.common.registration import RegistrationClient
    from infera.common.registration_k8s import K8sRegistrationClient

    for client in (RegistrationClient, K8sRegistrationClient):
        assert not hasattr(client, "announce_draining"), client.__name__


@pytest.mark.asyncio
async def test_the_k8s_registry_marks_a_condemned_pod_draining():
    """The one remaining producer of DRAINING, and the reason the worker does
    not need to announce anything under Kubernetes."""
    import json as _json

    from infera.common.discovery_k8s import WORKER_INFO_ANNOTATION, KubernetesRegistry

    reg = KubernetesRegistry(label_selector="x=y", namespace="ns")

    def _pod(*, deleting: bool):
        meta = {
            "name": "worker-1",
            "annotations": {WORKER_INFO_ANNOTATION: _json.dumps(build_worker_payload(_cfg()))},
        }
        if deleting:
            meta["deletionTimestamp"] = "2026-08-10T00:00:00Z"
        return {"metadata": meta, "status": {"phase": "Running"}}

    reg._handle_pod(_pod(deleting=False), deleted=False)
    assert [w.worker_id for w in reg._pool.list_active()] == ["10.0.0.1:8080"]

    reg._handle_pod(_pod(deleting=True), deleted=False)
    assert reg._pool.list_active() == [], "a condemned Pod must leave routing"
    assert reg._pool.get("10.0.0.1:8080").status is WorkerStatus.DRAINING
