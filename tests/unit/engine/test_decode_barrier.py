###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Prefill must not start SGLang PD warmup before a decode worker registers."""

from __future__ import annotations

import json

import httpx
import pytest

from infera.common.discovery_k8s import WORKER_INFO_ANNOTATION
from infera.engine.decode_barrier import (
    apply_pd_probe_recovery_defaults,
    decode_ready_timeout_seconds,
    ensure_skip_server_warmup,
    is_compatible_decode_worker,
    k8s_namespace,
    list_k8s_worker_payloads,
    resolve_k8s_label_selector,
    should_wait_for_decode,
    verify_pd_peer,
    wait_for_decode,
)


def _decode_payload(**overrides):
    payload = {
        "worker_id": "10.235.192.141:30000",
        "url": "http://10.235.192.141:30000",
        "model_name": "glm-5-3",
        "engine": "sglang",
        "disagg_mode": "decode",
        "disagg_meta": {"protocol": "sglang-bootstrap", "params": {}},
    }
    payload.update(overrides)
    return payload


def test_compatible_decode_worker_matches_sglang_bootstrap():
    assert is_compatible_decode_worker(_decode_payload(), model_name="glm-5-3")


@pytest.mark.parametrize(
    "overrides",
    [
        {"disagg_mode": "prefill"},
        {"disagg_mode": "mixed"},
        {"model_name": "other"},
        {"engine": "vllm"},
        {"disagg_meta": {}},
        {"disagg_meta": {"protocol": "vllm-mooncake"}},
    ],
)
def test_incompatible_decode_worker_is_rejected(overrides):
    assert not is_compatible_decode_worker(_decode_payload(**overrides), model_name="glm-5-3")


def test_should_wait_for_decode_defaults_on_for_prefill_only():
    assert should_wait_for_decode("prefill", None) is True
    assert should_wait_for_decode("prefill", True) is True
    assert should_wait_for_decode("prefill", False) is False
    assert should_wait_for_decode("decode", None) is False
    assert should_wait_for_decode("decode", True) is False
    assert should_wait_for_decode("mixed", None) is False


# --- scoping -----------------------------------------------------------------


def _clear_selector_env(monkeypatch):
    monkeypatch.delenv("INFERA_K8S_LABEL_SELECTOR", raising=False)
    monkeypatch.delenv("WORKLOAD_ID", raising=False)
    monkeypatch.delenv("POD_NAME", raising=False)


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_prefers_explicit_then_env(monkeypatch):
    _clear_selector_env(monkeypatch)
    assert await resolve_k8s_label_selector("app=x") == "app=x"
    monkeypatch.setenv("INFERA_K8S_LABEL_SELECTOR", "app=env")
    assert await resolve_k8s_label_selector(None) == "app=env"


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_reads_own_pod_label(monkeypatch):
    _clear_selector_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/namespaces/ns0/pods/prefill-0"
        return httpx.Response(
            200,
            json={"metadata": {"labels": {"infera.amd.com/deployment": "idep-a"}}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://k8s") as client:
        got = await resolve_k8s_label_selector(
            None, namespace="ns0", pod_name="prefill-0", http=client
        )
    assert got == "infera.amd.com/deployment=idep-a"


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_retries_own_pod_get(monkeypatch):
    _clear_selector_env(monkeypatch)
    monkeypatch.setenv("WORKLOAD_ID", "wrong-deployment")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="unavailable")
        return httpx.Response(
            200,
            json={"metadata": {"labels": {"infera.amd.com/deployment": "idep-a"}}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://k8s") as client:
        got = await resolve_k8s_label_selector(
            None,
            namespace="ns0",
            pod_name="prefill-0",
            http=client,
            retry_sleep=0.0,
        )
    assert got == "infera.amd.com/deployment=idep-a"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_does_not_use_workload_id_after_get_error(
    monkeypatch,
):
    """A blip talking to the apiserver must not silently pick another deployment."""
    _clear_selector_env(monkeypatch)
    monkeypatch.setenv("WORKLOAD_ID", "wrong-deployment")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="unavailable")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://k8s") as client:
        with pytest.raises(RuntimeError, match="could not read this Pod's labels"):
            await resolve_k8s_label_selector(
                None,
                namespace="ns0",
                pod_name="prefill-0",
                http=client,
                retries=2,
                retry_sleep=0.0,
            )


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_falls_back_to_workload_id(monkeypatch):
    _clear_selector_env(monkeypatch)
    monkeypatch.setenv("WORKLOAD_ID", "infera-glm53-1p1d-fhl7t")
    got = await resolve_k8s_label_selector(None, namespace="ns0", pod_name="")
    assert got == "infera.amd.com/deployment=infera-glm53-1p1d-fhl7t"


@pytest.mark.asyncio
async def test_resolve_k8s_label_selector_refuses_to_run_unscoped(monkeypatch):
    """An unscoped list would accept another deployment's decode worker."""
    _clear_selector_env(monkeypatch)
    with pytest.raises(RuntimeError, match="cannot scope the decode barrier"):
        await resolve_k8s_label_selector(None, namespace="ns0", pod_name="")


def test_k8s_namespace_prefers_pod_namespace(monkeypatch):
    monkeypatch.setenv("POD_NAMESPACE", "from-env")
    assert k8s_namespace() == "from-env"
    assert k8s_namespace("explicit") == "explicit"


# --- budget ------------------------------------------------------------------


def test_decode_ready_timeout_prefers_explicit_then_own_env(monkeypatch):
    monkeypatch.delenv("INFERA_DECODE_READY_TIMEOUT", raising=False)
    assert decode_ready_timeout_seconds(None) == 14400.0
    monkeypatch.setenv("INFERA_DECODE_READY_TIMEOUT", "60")
    assert decode_ready_timeout_seconds(None) == 60.0
    assert decode_ready_timeout_seconds(12.5) == 12.5


def test_decode_ready_timeout_ignores_the_engine_ready_timeout(monkeypatch):
    """That env is the engine's own /health budget; recipes retune it for slow
    weight loads, which must not silently move this barrier."""
    monkeypatch.delenv("INFERA_DECODE_READY_TIMEOUT", raising=False)
    monkeypatch.setenv("INFERA_ENGINE_READY_TIMEOUT", "10800")
    assert decode_ready_timeout_seconds(None) == 14400.0


def test_decode_ready_timeout_survives_a_malformed_value(monkeypatch):
    monkeypatch.setenv("INFERA_DECODE_READY_TIMEOUT", "later")
    assert decode_ready_timeout_seconds(None) == 14400.0


# --- polling -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_decode_returns_when_decode_registers():
    calls = {"n": 0}

    async def list_workers():
        calls["n"] += 1
        if calls["n"] < 2:
            return [{"disagg_mode": "prefill", "model_name": "glm-5-3"}]
        return [_decode_payload()]

    found = await wait_for_decode(
        list_workers,
        model_name="glm-5-3",
        timeout=2.0,
        poll_interval=0.01,
    )
    assert found["worker_id"] == "10.235.192.141:30000"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_wait_for_decode_retries_a_failed_lookup():
    """A transient apiserver/etcd error must not kill prefill before it starts."""
    calls = {"n": 0}

    async def list_workers():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("apiserver unreachable")
        return [_decode_payload()]

    found = await wait_for_decode(
        list_workers,
        model_name="glm-5-3",
        timeout=2.0,
        poll_interval=0.01,
    )
    assert found["worker_id"] == "10.235.192.141:30000"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_for_decode_times_out_without_decode():
    async def list_workers():
        return []

    with pytest.raises(TimeoutError, match="poison Mooncake"):
        await wait_for_decode(
            list_workers,
            model_name="glm-5-3",
            timeout=0.05,
            poll_interval=0.01,
        )


@pytest.mark.asyncio
async def test_wait_for_decode_times_out_when_every_lookup_fails():
    async def list_workers():
        raise httpx.ConnectError("apiserver unreachable")

    with pytest.raises(TimeoutError):
        await wait_for_decode(
            list_workers,
            model_name="glm-5-3",
            timeout=0.05,
            poll_interval=0.01,
        )


# --- Pod filtering -----------------------------------------------------------


def _pod(name, payload=None, *, phase="Running", terminating=False, ready=True):
    meta = {"name": name, "annotations": {}}
    if terminating:
        meta["deletionTimestamp"] = "2026-09-17T00:00:00Z"
    if payload is not None:
        meta["annotations"][WORKER_INFO_ANNOTATION] = json.dumps(payload)
    status = {
        "phase": phase,
        "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
    }
    return {"metadata": meta, "status": status}


@pytest.mark.asyncio
async def test_list_k8s_worker_payloads_keeps_only_live_registered_peers():
    items = [
        _pod("prefill-self", _decode_payload()),
        _pod("decode-ok", _decode_payload()),
        _pod("decode-term", _decode_payload(), terminating=True),
        _pod("pending", _decode_payload(), phase="Pending"),
        # Restarted: the annotation outlives the process that wrote it, so the
        # readiness gate is the only thing separating this from a live peer.
        _pod("decode-restarting", _decode_payload(), ready=False),
        _pod("no-ann"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["labelSelector"] == "infera.amd.com/deployment=x"
        return httpx.Response(200, json={"items": items})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://k8s") as client:
        found = await list_k8s_worker_payloads(
            namespace="default-default",
            label_selector="infera.amd.com/deployment=x",
            skip_pod_name="prefill-self",
            http=client,
        )
    assert [p["worker_id"] for p in found] == ["10.235.192.141:30000"]


def test_ensure_skip_server_warmup_is_idempotent():
    assert ensure_skip_server_warmup(["--tp-size", "8"]) == [
        "--tp-size",
        "8",
        "--skip-server-warmup",
    ]
    already = ["--skip-server-warmup", "--tp-size", "8"]
    assert ensure_skip_server_warmup(already) is already


@pytest.mark.asyncio
async def test_verify_pd_peer_transfers_real_kv_before_registration():
    requests: list[tuple[str, dict]] = []
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((f"{request.url.host}{request.url.path}", body))
        return httpx.Response(200, json={"text": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await verify_pd_peer(
            prefill_url="http://prefill:30000",
            decode_url="http://decode:30000",
            bootstrap_host="prefill",
            bootstrap_port=30001,
            room_seed=100,
            http=client,
            sleep=fake_sleep,
        )

    assert sleeps == []

    assert [path for path, _ in requests] == [
        "prefill/generate",
        "decode/generate",
    ]
    assert requests[0][1] == requests[1][1]
    assert requests[0][1]["bootstrap_host"] == "prefill"
    assert requests[0][1]["bootstrap_port"] == 30001
    assert requests[0][1]["bootstrap_room"] == 100
    assert requests[0][1]["rid"] == "infera-probe-100"


@pytest.mark.asyncio
async def test_verify_pd_peer_maps_prefill_ranks_to_smaller_decode_dp():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        return httpx.Response(200, json={"text": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await verify_pd_peer(
            prefill_url="http://prefill:30000",
            decode_url="http://decode:30000",
            bootstrap_host="prefill",
            bootstrap_port=30001,
            dp_size=4,
            decode_dp_size=1,
            room_seed=100,
            http=client,
        )

    prefill = [body for host, body in requests if host == "prefill"]
    decode = [body for host, body in requests if host == "decode"]
    assert [body["routed_dp_rank"] for body in prefill] == [0, 1, 2, 3]
    assert [body["routed_dp_rank"] for body in decode] == [0, 0, 0, 0]
    assert [body["bootstrap_room"] % 4 for body in prefill] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_verify_pd_peer_aborts_both_legs_on_transfer_failure():
    aborts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            aborts.append(request.url.host)
            return httpx.Response(200)
        if request.url.host == "decode":
            return httpx.Response(500, text="KVTransferError")
        return httpx.Response(200, json={"text": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="PD peer verification failed"):
            await verify_pd_peer(
                prefill_url="http://prefill:30000",
                decode_url="http://decode:30000",
                bootstrap_host="prefill",
                bootstrap_port=30001,
                room_seed=101,
                attempts=1,
                http=client,
            )

    assert sorted(aborts) == ["decode", "prefill"]


@pytest.mark.asyncio
async def test_verify_pd_peer_retries_after_abort_then_succeeds():
    generate_calls = {"n": 0}
    aborts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            aborts.append(request.url.host)
            return httpx.Response(200)
        generate_calls["n"] += 1
        if generate_calls["n"] <= 2:
            return httpx.Response(500, text="KVTransferError")
        return httpx.Response(200, json={"text": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await verify_pd_peer(
            prefill_url="http://prefill:30000",
            decode_url="http://decode:30000",
            bootstrap_host="prefill",
            bootstrap_port=30001,
            room_seed=200,
            attempts=3,
            retry_sleep=0.0,
            http=client,
        )

    assert generate_calls["n"] == 4
    assert sorted(aborts) == ["decode", "prefill"]


@pytest.mark.asyncio
async def test_verify_pd_peer_gives_up_after_retry_budget():
    generate_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            return httpx.Response(200)
        generate_calls["n"] += 1
        return httpx.Response(500, text="KVTransferError")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            await verify_pd_peer(
                prefill_url="http://prefill:30000",
                decode_url="http://decode:30000",
                bootstrap_host="prefill",
                bootstrap_port=30001,
                room_seed=300,
                attempts=3,
                retry_sleep=0.0,
                http=client,
            )

    assert generate_calls["n"] == 6


def test_apply_pd_probe_recovery_defaults_for_mooncake_prefill(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_FAILED_SESSION_PROBE", raising=False)
    monkeypatch.delenv("SGLANG_FAILED_SESSION_PROBE_INTERVAL_S", raising=False)

    applied = apply_pd_probe_recovery_defaults("prefill", "mooncake")

    assert applied == {
        "SGLANG_ENABLE_FAILED_SESSION_PROBE": "1",
        "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S": "5",
    }


def test_apply_pd_probe_recovery_defaults_preserves_overrides(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_FAILED_SESSION_PROBE", "0")
    monkeypatch.setenv("SGLANG_FAILED_SESSION_PROBE_INTERVAL_S", "12")

    applied = apply_pd_probe_recovery_defaults("prefill", "mooncake")

    assert applied == {}


@pytest.mark.asyncio
async def test_verify_pd_peer_waits_between_retries_for_rdma_recovery():
    sleeps: list[float] = []
    generate_calls = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            return httpx.Response(200)
        generate_calls["n"] += 1
        if generate_calls["n"] <= 2:
            return httpx.Response(500, text="KVTransferError")
        return httpx.Response(200, json={"text": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await verify_pd_peer(
            prefill_url="http://prefill:30000",
            decode_url="http://decode:30000",
            bootstrap_host="prefill",
            bootstrap_port=30001,
            room_seed=400,
            attempts=3,
            http=client,
            sleep=fake_sleep,
        )

    assert sleeps == [35.0]
