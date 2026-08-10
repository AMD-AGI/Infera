###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The shutdown sequence: deregister, then drain.

Removing the record is what stops new work arriving, so it has to happen before
waiting on work already in flight. These pin that order in the entrypoints
themselves and pin that a deregistration which did not happen says so.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from infera.common.registration import RegistrationClient
from infera.common.registration_k8s import K8sRegistrationClient

ENTRYPOINTS = (
    "infera/engine/vllm/__main__.py",
    "infera/engine/sglang/__main__.py",
    "infera/tools/fakeworker/server.py",
)


def _shutdown_call_order(path: Path) -> list[str]:
    """The two calls that matter, in source order.

    Reduced to deregistering and draining, so reordering anything else in the
    shutdown sequence does not make this fail. Calling `_drain` is a Call node
    while defining it is not, so the definition is not counted.
    """
    seen = []
    for call in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)):
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "deregister":
            seen.append(("deregister", call.lineno))
        elif isinstance(func, ast.Name) and func.id == "_drain":
            seen.append(("drain", call.lineno))
    # ast.walk is breadth-first, so sort back into source order.
    return [name for name, _ in sorted(seen, key=lambda p: p[1])]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_every_entrypoint_deregisters_before_it_drains(entrypoint):
    """Draining while still registered means the router keeps assigning work
    for the whole drain window, so the drain never converges.

    On etcd removing the record is the only way to express departure. On
    Kubernetes the registry does drop a Pod on its deletionTimestamp, but only
    when the Pod is being deleted -- a liveness-probe restart, a node graceful
    shutdown or a manual kill all send SIGTERM with the Pod object untouched,
    leaving the annotation present and the worker routable until it clears it.
    """
    root = Path(inspect.getfile(RegistrationClient)).parents[2]
    order = _shutdown_call_order(root / entrypoint)

    assert "deregister" in order, f"{entrypoint} never deregisters"
    assert "drain" in order, f"{entrypoint} never drains"
    assert order.index("deregister") < order.index("drain"), (
        f"{entrypoint} drains before deregistering, so new work keeps arriving "
        "for the whole drain window"
    )


class _Resp:
    def __init__(self, code: int):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_etcd_reports_a_revoke_the_server_refused():
    """httpx does not raise on 4xx/5xx, so a refused revoke reaches the same
    code path as a successful one. etcd answering 'lease not found', or a proxy
    answering 503, is the likeliest way this fails."""

    class _Http:
        async def post(self, path, json=None):  # noqa: A002 - mirrors httpx
            return _Resp(500)

        async def aclose(self):
            pass

    # __new__, so the real httpx client is never built and never leaked.
    c = RegistrationClient.__new__(RegistrationClient)
    c._http = _Http()
    c._lease_id, c._key, c._worker_id, c._lease_ttl = 1, "/k", "w", 30

    assert await c.deregister() is False, (
        "etcd refused the revoke, so the lease is still alive and the worker is "
        "still routable -- that cannot report success"
    )


@pytest.mark.asyncio
async def test_etcd_reports_an_unreachable_server():
    class _Http:
        async def post(self, path, json=None):  # noqa: A002 - mirrors httpx
            raise RuntimeError("etcd unreachable")

        async def aclose(self):
            pass

    # __new__, so the real httpx client is never built and never leaked.
    c = RegistrationClient.__new__(RegistrationClient)
    c._http = _Http()
    c._lease_id, c._key, c._worker_id, c._lease_ttl = 1, "/k", "w", 30

    assert await c.deregister() is False


@pytest.mark.asyncio
async def test_kubernetes_reports_a_patch_that_failed():
    """Clearing the annotation is what takes the worker out of the pool on the
    paths where the Pod object is untouched."""
    c = K8sRegistrationClient.__new__(K8sRegistrationClient)
    c._worker_id = "w"
    c._pod_name = "p"
    c._namespace = "ns"

    async def _patch(*_args, **_kwargs):
        raise RuntimeError("apiserver unreachable")

    c._patch_annotation = _patch

    assert await c.deregister() is False
