###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Actuation: what each connector actually puts on the wire."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from infera.planner.args import PlannerArgs
from infera.planner.connectors.base import (
    NoOperationConnector,
    PlannerConnector,
    build_connector,
)
from infera.planner.connectors.kubernetes import KubernetesConnector
from infera.planner.connectors.virtual import VirtualConnector
from infera.planner.decision import ScalingDecision


def decision(prefill: int = 3, decode: int = 5, **overrides) -> ScalingDecision:
    return ScalingDecision(
        num_prefill=prefill,
        num_decode=decode,
        observed_prefill=1,
        observed_decode=1,
        **overrides,
    )


def recording_client(status: int = 200, body: str = "{}") -> tuple[httpx.AsyncClient, list]:
    """An httpx client that records every request instead of sending it.

    Carries a ``base_url`` because both connectors address their endpoints with
    relative paths, exactly as the real clients are constructed.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, text=body)

    client = httpx.AsyncClient(
        base_url="https://control-plane.invalid",
        transport=httpx.MockTransport(handler),
    )
    return client, seen


class TestVirtualConnector:
    def test_key_is_derived_from_the_prefix(self):
        assert (
            VirtualConnector(etcd_endpoint="127.0.0.1:2379", prefix="/infera/workers").key
            == "/infera/workers/planner/decision"
        )

    def test_a_prefix_with_a_trailing_slash_is_not_doubled(self):
        assert (
            VirtualConnector(etcd_endpoint="127.0.0.1:2379", prefix="/infera/workers/").key
            == "/infera/workers/planner/decision"
        )

    def test_payload_carries_absolute_counts(self):
        # Absolute rather than deltas, so a consumer that misses a decision has
        # nothing to catch up on -- applying the newest is always correct.
        conn = VirtualConnector(etcd_endpoint="127.0.0.1:2379")
        payload = conn.payload_for(decision(prefill=3, decode=5), 7)
        assert payload["num_prefill_workers"] == 3
        assert payload["num_decode_workers"] == 5
        assert payload["decision_id"] == 7
        assert payload["observed_prefill"] == 1
        assert payload["gpu_budget_exceeded"] is False

    async def test_writes_a_base64_kv_put_to_etcd(self):
        conn = VirtualConnector(etcd_endpoint="127.0.0.1:2379")
        conn._http, seen = recording_client()
        await conn.apply(decision(prefill=2, decode=6))

        assert len(seen) == 1
        assert seen[0].url.path == "/v3/kv/put"
        sent = json.loads(seen[0].content)
        assert base64.b64decode(sent["key"]).decode() == "/infera/workers/planner/decision"
        value = json.loads(base64.b64decode(sent["value"]))
        assert value["num_prefill_workers"] == 2
        assert value["num_decode_workers"] == 6
        await conn.aclose()

    async def test_decision_id_increments_per_write(self):
        # A consumer uses this to tell a fresh decision from a re-read.
        conn = VirtualConnector(etcd_endpoint="127.0.0.1:2379")
        conn._http, seen = recording_client()
        await conn.apply(decision())
        await conn.apply(decision())
        ids = [
            json.loads(base64.b64decode(json.loads(r.content)["value"]))["decision_id"]
            for r in seen
        ]
        assert ids == [1, 2]
        await conn.aclose()

    async def test_a_failed_write_still_advances_the_decision_id(self):
        # The next successful write must look newer than anything a consumer may
        # already have seen, so the id cannot be reused.
        conn = VirtualConnector(etcd_endpoint="127.0.0.1:2379")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("etcd down", request=request)

        conn._http = httpx.AsyncClient(
            base_url="https://etcd.invalid", transport=httpx.MockTransport(handler)
        )
        await conn.apply(decision())  # swallowed, not raised
        assert conn._decision_id == 1
        await conn.aclose()

    async def test_etcd_rejecting_the_write_is_not_fatal(self):
        conn = VirtualConnector(etcd_endpoint="127.0.0.1:2379")
        conn._http, _ = recording_client(status=500, body="boom")
        await conn.apply(decision())  # must not raise; the next interval retries
        await conn.aclose()


class TestKubernetesConnector:
    def _connector(self, **overrides) -> KubernetesConnector:
        kwargs = {"deployment_name": "infera-pd", "namespace": "serving"}
        kwargs.update(overrides)
        conn = KubernetesConnector.__new__(KubernetesConnector)
        conn.deployment_name = kwargs["deployment_name"]
        conn.namespace = kwargs["namespace"]
        conn.prefill_service = kwargs.get("prefill_service", "prefill")
        conn.decode_service = kwargs.get("decode_service", "decode")
        return conn

    def test_resource_path_addresses_the_custom_resource(self):
        assert self._connector().resource_path == (
            "/apis/infera.amd.com/v1alpha1/namespaces/serving/inferadeployments/infera-pd"
        )

    def test_patch_touches_only_the_replica_fields(self):
        # A merge patch this narrow cannot clobber a concurrent edit to the rest
        # of the spec.
        patch = self._connector().patch_for(decision(prefill=3, decode=5))
        assert patch == {
            "spec": {"services": {"prefill": {"replicas": 3}, "decode": {"replicas": 5}}}
        }

    def test_service_names_are_configurable(self):
        patch = self._connector(prefill_service="p-workers", decode_service="d-workers").patch_for(
            decision(prefill=2, decode=4)
        )
        assert patch["spec"]["services"]["p-workers"]["replicas"] == 2
        assert patch["spec"]["services"]["d-workers"]["replicas"] == 4

    def test_deployment_name_is_required(self):
        with pytest.raises(ValueError, match="InferaDeployment name"):
            KubernetesConnector(deployment_name="")

    async def test_sends_a_merge_patch(self):
        conn = self._connector()
        conn._http, seen = recording_client()
        await conn.apply(decision(prefill=3, decode=5))

        assert len(seen) == 1
        assert seen[0].method == "PATCH"
        assert seen[0].headers["Content-Type"] == "application/merge-patch+json"
        assert seen[0].url.path.endswith("/inferadeployments/infera-pd")
        assert json.loads(seen[0].content)["spec"]["services"]["decode"]["replicas"] == 5
        await conn.aclose()

    @pytest.mark.parametrize("status", [403, 404, 500])
    async def test_an_api_server_rejection_is_logged_not_raised(self, status):
        # 404 means the deployment name or namespace is wrong, 403 means missing
        # RBAC. Neither should take the planner down; the next interval retries.
        conn = self._connector()
        conn._http, _ = recording_client(status=status, body="nope")
        await conn.apply(decision())
        await conn.aclose()

    async def test_an_unreachable_api_server_is_not_fatal(self):
        conn = self._connector()

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        conn._http = httpx.AsyncClient(
            base_url="https://kubernetes.invalid", transport=httpx.MockTransport(handler)
        )
        await conn.apply(decision())
        await conn.aclose()


class TestBuildConnector:
    def _args(self, **overrides) -> PlannerArgs:
        defaults = {"profile_results": "x", "connector": "virtual"}
        defaults.update(overrides)
        return PlannerArgs(**defaults)

    def test_no_operation_overrides_the_chosen_backend(self):
        # --no-operation must win even when a real connector is configured,
        # otherwise a dry run would still resize the deployment.
        conn = build_connector(
            self._args(connector="kubernetes", deployment_name="infera-pd", no_operation=True)
        )
        assert isinstance(conn, NoOperationConnector)

    def test_builds_the_virtual_connector(self):
        conn = build_connector(self._args(etcd_prefix="/custom"))
        assert isinstance(conn, VirtualConnector)
        assert conn.key == "/custom/planner/decision"

    def test_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown connector"):
            build_connector(self._args(connector="carrier-pigeon"))

    @pytest.mark.parametrize(
        "connector",
        [NoOperationConnector(), VirtualConnector(etcd_endpoint="127.0.0.1:2379")],
    )
    def test_every_connector_satisfies_the_protocol(self, connector):
        assert isinstance(connector, PlannerConnector)

    async def test_no_operation_connector_does_nothing(self):
        conn = NoOperationConnector()
        await conn.apply(decision())
        await conn.aclose()
