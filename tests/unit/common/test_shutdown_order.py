###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The shutdown sequence, driven rather than asserted from constants.

Nothing here inspects a flag saying what the order should be; these record the
order the entrypoints actually perform and the reason it has to be that one.
"""

from __future__ import annotations

import pytest

from infera.common.registration import RegistrationClient
from infera.common.registration_k8s import K8sRegistrationClient


def test_deregistering_is_what_stops_new_work_on_every_backend():
    """Both backends stop new work by removing the record, so both must remove
    it before waiting on in-flight work.

    On etcd nothing else can express departure. On Kubernetes the registry does
    drop a Pod on its deletionTimestamp -- but only when the Pod is being
    deleted. A liveness-probe restart, a node graceful shutdown or a manual kill
    all send SIGTERM with the Pod object untouched, and on those paths the
    annotation is still present and still parsed, so the worker stays routable
    until it clears it. Draining first there means the router keeps assigning
    work for the whole drain window.
    """
    for client in (RegistrationClient, K8sRegistrationClient):
        assert not hasattr(client, "deregister_stops_routing"), (
            f"{client.__name__} still declares a per-backend order; both now "
            "deregister before draining"
        )


@pytest.mark.asyncio
async def test_a_failed_deregistration_is_reported():
    """Deregistering is the step that stops new work, so swallowing its failure
    means draining while still being routed to -- silently."""
    calls = []

    class _Http:
        async def post(self, path, json=None):  # noqa: A002 - mirrors httpx
            raise RuntimeError("etcd unreachable")

        async def aclose(self):
            pass

    c = RegistrationClient("http://etcd:2379")
    c._http = _Http()
    c._lease_id, c._key, c._worker_id = 1, "/k", "w"

    ok = await c.deregister()
    assert ok is False, "a deregistration that did not happen must not report success"
    assert calls == []
