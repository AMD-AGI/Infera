###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Resizing pools through the router.

The write itself is one PATCH, so most of what matters here is what never
reaches the cluster: a request that would empty a pool, name one that does not
exist, or apply to some services and not others.
"""

from __future__ import annotations

import json

import pytest

from infera.server.scaling import DeploymentScaler, ScalingError

POD = {"metadata": {"labels": {"infera.amd.com/deployment": "qwen"}}}

CR = {
    "spec": {
        "services": {
            "server": {"componentType": "server", "replicas": 1},
            "prefill": {"componentType": "worker", "role": "prefill", "replicas": 2},
            "decode": {"componentType": "worker", "role": "decode", "replicas": 4},
        }
    },
    "status": {
        "state": "ready",
        "services": {
            "server": {"replicas": 1, "readyReplicas": 1},
            "prefill": {"replicas": 2, "readyReplicas": 2},
            "decode": {"replicas": 4, "readyReplicas": 3},
        },
    },
}


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected {self.status_code}")


class _FakeApi:
    """Stands in for the API server, recording what was written to it."""

    def __init__(self, *, pod=POD, cr=None, get_status=200, patch_status=200, pods=None):
        self.pod = pod
        self.pods = pods or []
        self.cr = json.loads(json.dumps(cr if cr is not None else CR))
        self.get_status = get_status
        self.patch_status = patch_status
        self.patches: list[dict] = []

    async def get(self, path, params=None):
        if path.endswith("/pods"):
            return _Resp(200, {"items": self.pods})
        if "/pods/" in path:
            return _Resp(200 if self.pod else 404, self.pod)
        if self.get_status != 200:
            return _Resp(self.get_status)
        return _Resp(200, self.cr)

    async def patch(self, path, content=None, headers=None):
        if self.patch_status != 200:
            return _Resp(self.patch_status)
        body = json.loads(content)
        self.patches.append(body)
        # Reflect the write, so the snapshot returned afterwards is the new state.
        for svc, cfg in body["spec"]["services"].items():
            self.cr["spec"]["services"][svc]["replicas"] = cfg["replicas"]
        return _Resp(200, self.cr)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def api(monkeypatch):
    fake = _FakeApi()
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)
    monkeypatch.setattr("infera.server.scaling.in_cluster_namespace", lambda *a, **k: "infera")
    return fake


def _scaler():
    return DeploymentScaler(namespace="infera", pod_name="router-abc")


@pytest.mark.asyncio
async def test_a_snapshot_reports_both_asked_for_and_achieved(api):
    """A caller deciding whether to scale again needs to know whether the last
    request has landed."""
    snap = await _scaler().snapshot()

    assert snap["deployment"] == "qwen"
    assert snap["services"]["decode"] == {
        "role": "decode",
        "replicas": 4,
        "current_replicas": 4,
        "ready_replicas": 3,
        "nodes_per_replica": 1,
    }


@pytest.mark.asyncio
async def test_prefill_and_decode_move_in_one_write(api):
    """Rebalancing one against the other must not be able to half-apply: a PD
    deployment left lopsided serves badly until someone notices."""
    await _scaler().scale({"prefill": 4, "decode": 8})

    assert len(api.patches) == 1, "a partial application must not be possible"
    assert api.patches[0] == {
        "spec": {"services": {"prefill": {"replicas": 4}, "decode": {"replicas": 8}}}
    }


@pytest.mark.asyncio
async def test_only_the_named_services_are_touched(api):
    await _scaler().scale({"decode": 6})

    written = api.patches[0]["spec"]["services"]
    assert set(written) == {"decode"}
    assert api.cr["spec"]["services"]["prefill"]["replicas"] == 2


@pytest.mark.asyncio
async def test_the_deployment_is_found_from_the_routers_own_pod(api):
    """Configuring the name separately would let the two disagree -- and a stale
    one would resize a fleet nobody asked about."""
    await _scaler().scale({"decode": 5})
    assert api.cr["spec"]["services"]["decode"]["replicas"] == 5


@pytest.mark.asyncio
async def test_an_unknown_service_is_refused_before_anything_is_written(api):
    with pytest.raises(ScalingError) as err:
        await _scaler().scale({"decode": 5, "prefil": 2})  # typo

    assert "prefil" in str(err.value)
    assert "prefill" in str(err.value), "the message should say what does exist"
    assert api.patches == [], "a rejected request must not partially apply"


@pytest.mark.asyncio
async def test_scaling_to_zero_is_refused(api):
    """An empty pool stops serving, and in PD an empty side fails every request
    rather than only the ones that would have landed there."""
    with pytest.raises(ScalingError) as err:
        await _scaler().scale({"decode": 0})

    assert "stops serving" in str(err.value)
    assert api.patches == []


@pytest.mark.asyncio
async def test_a_negative_count_is_refused(api):
    with pytest.raises(ScalingError):
        await _scaler().scale({"decode": -1})
    assert api.patches == []


@pytest.mark.asyncio
async def test_a_non_integer_count_is_refused(api):
    for bad in ("4", 4.5, None, True):
        with pytest.raises(ScalingError):
            await _scaler().scale({"decode": bad})
    assert api.patches == []


@pytest.mark.asyncio
async def test_an_empty_request_is_refused(api):
    with pytest.raises(ScalingError):
        await _scaler().scale({})


@pytest.mark.asyncio
async def test_a_router_the_operator_did_not_create_says_so(monkeypatch):
    """Deployed by hand, there is no deployment to resize -- and no CR that
    would be the right one to guess at."""
    fake = _FakeApi(pod={"metadata": {"labels": {}}})
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    with pytest.raises(ScalingError) as err:
        await _scaler().snapshot()

    assert err.value.status == 409
    assert "not created by the operator" in str(err.value)


@pytest.mark.asyncio
async def test_a_missing_permission_names_the_fix(monkeypatch):
    """The RBAC for this ships with the operator, so an older deployment has a
    router that cannot write. That is a setup step, not a bug to debug."""
    fake = _FakeApi(get_status=403)
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    with pytest.raises(ScalingError) as err:
        await _scaler().snapshot()

    assert err.value.status == 403
    assert "re-apply the operator RBAC" in str(err.value)


@pytest.mark.asyncio
async def test_a_missing_deployment_is_reported_as_such(monkeypatch):
    fake = _FakeApi(get_status=404)
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    with pytest.raises(ScalingError) as err:
        await _scaler().snapshot()
    assert err.value.status == 404


# ------------------------------------------------------------------
# Telling "still starting" from "never will"
# ------------------------------------------------------------------


def pending_pod(reason, message, *, kind="scheduling"):
    if kind == "scheduling":
        return {
            "status": {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": reason,
                        "message": message,
                    }
                ],
            }
        }
    return {
        "status": {
            "phase": "Pending",
            "conditions": [{"type": "PodScheduled", "status": "True"}],
            "containerStatuses": [{"state": {"waiting": {"reason": reason, "message": message}}}],
        }
    }


def short_cr(want=8, ready=3):
    """A pool the cluster could not fill: the replicas that were never created
    are missing from the status, so the two counts agree with each other and
    disagree with the spec."""
    return {
        "spec": {"services": {"decode": {"role": "decode", "replicas": want}}},
        "status": {
            "state": "pending",
            "services": {"decode": {"replicas": ready, "readyReplicas": ready}},
        },
    }


@pytest.mark.asyncio
async def test_the_operators_own_verdict_is_reported(api):
    """It compares ready replicas against the spec, which is the comparison a
    caller wants and the easy one to get wrong."""
    assert (await _scaler().snapshot())["state"] == "ready"


@pytest.mark.asyncio
async def test_a_pool_the_scheduler_could_not_place_says_so(monkeypatch):
    """The counts alone cannot distinguish this from a slow start: a replica
    that was never created is absent from both of them."""
    fake = _FakeApi(
        cr=short_cr(),
        pods=[
            pending_pod("Unschedulable", "0/13 nodes are available: 8 Insufficient amd.com/gpu.")
        ],
    )
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    pool = (await _scaler().snapshot())["services"]["decode"]

    assert pool["replicas"] == 8
    assert pool["ready_replicas"] == 3
    assert "Insufficient amd.com/gpu" in pool["blocked"]
    assert pool["blocked"].startswith("Unschedulable:")


@pytest.mark.asyncio
async def test_a_pod_that_will_never_start_is_reported_too(monkeypatch):
    """Scheduled but stuck reads the same as starting, from the counts."""
    fake = _FakeApi(
        cr=short_cr(),
        pods=[pending_pod("ImagePullBackOff", "Back-off pulling image", kind="waiting")],
    )
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    blocked = (await _scaler().snapshot())["services"]["decode"]["blocked"]
    assert blocked.startswith("ImagePullBackOff:")


@pytest.mark.asyncio
async def test_a_pod_merely_starting_is_not_reported_as_blocked(monkeypatch):
    """ContainerCreating resolves in seconds. Reporting it would make every
    scale-up look stuck for as long as a model takes to load."""
    fake = _FakeApi(
        cr=short_cr(),
        pods=[pending_pod("ContainerCreating", "", kind="waiting")],
    )
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    assert "blocked" not in (await _scaler().snapshot())["services"]["decode"]


@pytest.mark.asyncio
async def test_a_pool_that_is_up_is_not_investigated(monkeypatch):
    """No gap, no Pod query: the common path should not pay for the rare one."""
    fake = _FakeApi(
        cr=short_cr(want=3, ready=3),
        pods=[pending_pod("Unschedulable", "should not be read")],
    )
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    assert "blocked" not in (await _scaler().snapshot())["services"]["decode"]


@pytest.mark.asyncio
async def test_an_unreadable_pod_list_does_not_hide_the_counts(monkeypatch):
    """The reason is a convenience; the numbers are the answer."""

    class _NoPods(_FakeApi):
        async def get(self, path, params=None):
            if path.endswith("/pods"):
                raise RuntimeError("forbidden")
            return await super().get(path, params)

    fake = _NoPods(cr=short_cr())
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    pool = (await _scaler().snapshot())["services"]["decode"]
    assert pool["ready_replicas"] == 3
    assert "blocked" not in pool


@pytest.mark.asyncio
async def test_a_long_scheduler_message_is_cut_to_one_line(monkeypatch):
    """Scheduler messages enumerate every node; the head carries the verdict."""
    fake = _FakeApi(
        cr=short_cr(),
        pods=[pending_pod("Unschedulable", "node-a: no gpu.\n" + "x" * 900)],
    )
    monkeypatch.setattr("infera.server.scaling.make_client", lambda **kw: fake)

    blocked = (await _scaler().snapshot())["services"]["decode"]["blocked"]
    assert len(blocked) < 350
    assert "\n" not in blocked


# ------------------------------------------------------------------
# The HTTP surface
# ------------------------------------------------------------------


def _client(scaler):
    from fastapi.testclient import TestClient

    from infera.server import app as app_module

    app_module._scaler = scaler
    return TestClient(app_module.app)


def test_the_endpoints_are_off_unless_enabled():
    """It writes to the cluster and /v1/admin has no authentication of its own,
    so an operator has to ask for it."""
    c = _client(None)
    assert c.get("/v1/admin/scale").status_code == 403
    assert c.post("/v1/admin/scale", json={"services": {"decode": 2}}).status_code == 403
    assert "--enable-scaling-api" in c.get("/v1/admin/scale").json()["detail"]


def test_a_scale_request_reports_the_state_it_produced(api):
    c = _client(_scaler())
    resp = c.post("/v1/admin/scale", json={"services": {"decode": {"replicas": 6}}})

    assert resp.status_code == 200
    assert resp.json()["services"]["decode"]["replicas"] == 6
    assert api.patches[0]["spec"]["services"] == {"decode": {"replicas": 6}}


def test_a_bare_number_is_accepted_like_the_nested_form(api):
    """The nested form matches the CR, so a caller can paste between the two;
    the bare number is what anyone writes by hand."""
    c = _client(_scaler())
    assert c.post("/v1/admin/scale", json={"services": {"decode": 3}}).status_code == 200
    assert api.patches[0]["spec"]["services"] == {"decode": {"replicas": 3}}


def test_a_refused_request_says_why(api):
    c = _client(_scaler())
    resp = c.post("/v1/admin/scale", json={"services": {"decode": 0}})

    assert resp.status_code == 400
    assert "stops serving" in resp.json()["detail"]
    assert api.patches == []


def test_a_malformed_body_is_rejected(api):
    c = _client(_scaler())
    assert c.post("/v1/admin/scale", json={"decode": 3}).status_code == 400
    assert c.post("/v1/admin/scale", json={"services": 3}).status_code == 400
    assert api.patches == []
