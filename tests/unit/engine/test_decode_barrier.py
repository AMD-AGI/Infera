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
    decode_ready_timeout_seconds,
    is_compatible_decode_worker,
    list_k8s_worker_payloads,
    resolve_k8s_label_selector,
    should_wait_for_decode,
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


def test_resolve_k8s_label_selector_uses_workload_id(monkeypatch):
    monkeypatch.delenv("INFERA_K8S_LABEL_SELECTOR", raising=False)
    monkeypatch.setenv("WORKLOAD_ID", "infera-glm53-1p1d-fhl7t")
    assert resolve_k8s_label_selector(None) == "infera.amd.com/deployment=infera-glm53-1p1d-fhl7t"
    assert resolve_k8s_label_selector("app=x") == "app=x"


def test_decode_ready_timeout_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("INFERA_DECODE_READY_TIMEOUT", raising=False)
    monkeypatch.delenv("INFERA_ENGINE_READY_TIMEOUT", raising=False)
    assert decode_ready_timeout_seconds(None) == 14400.0
    monkeypatch.setenv("INFERA_ENGINE_READY_TIMEOUT", "10800")
    assert decode_ready_timeout_seconds(None) == 10800.0
    monkeypatch.setenv("INFERA_DECODE_READY_TIMEOUT", "60")
    assert decode_ready_timeout_seconds(None) == 60.0
    assert decode_ready_timeout_seconds(12.5) == 12.5


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


def _pod(name, payload=None, *, phase="Running", terminating=False):
    meta = {"name": name, "annotations": {}}
    if terminating:
        meta["deletionTimestamp"] = "2026-09-17T00:00:00Z"
    if payload is not None:
        meta["annotations"][WORKER_INFO_ANNOTATION] = json.dumps(payload)
    return {"metadata": meta, "status": {"phase": phase}}


@pytest.mark.asyncio
async def test_list_k8s_worker_payloads_skips_self_and_terminating():
    items = [
        _pod("prefill-self", _decode_payload()),
        _pod("decode-ok", _decode_payload()),
        _pod("decode-term", _decode_payload(), terminating=True),
        _pod("pending", _decode_payload(), phase="Pending"),
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
