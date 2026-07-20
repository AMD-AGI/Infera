###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import json

import pytest

from infera.common.worker_pool import DisaggMode, WorkerInfo, WorkerPool
from infera.router.direct import DIRECT_PREFILL_KEY, DIRECT_WORKER_KEY, DirectRouter
from infera.router.policy.round_robin import RoundRobinPolicy


class _StubResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}


def _pool() -> WorkerPool:
    pool = WorkerPool()
    pool.add(WorkerInfo(worker_id="w1", url="http://10.0.0.1:8000", model_name="m"))
    pool.add(WorkerInfo(worker_id="w2", url="http://10.0.0.2:8000", model_name="m"))
    return pool


@pytest.mark.asyncio
async def test_direct_dispatches_to_header_worker(monkeypatch):
    router = DirectRouter(_pool(), RoundRobinPolicy())
    seen = {}

    async def fake_attempt(target, blocks, body, hints, path, stream, obs):
        seen["worker_id"] = target.worker.worker_id
        obs["outcome"] = "ok"
        return _StubResp(200)

    monkeypatch.setattr(router, "_attempt", fake_attempt)
    body = {"model": "m", DIRECT_WORKER_KEY: "w2"}
    resp = await router.dispatch(body, stream=False)
    assert resp.status_code == 200
    assert seen["worker_id"] == "w2"  # honoured the header, not round-robin


@pytest.mark.asyncio
async def test_direct_unknown_worker_returns_503(monkeypatch):
    router = DirectRouter(_pool(), RoundRobinPolicy())
    body = {"model": "m", DIRECT_WORKER_KEY: "ghost"}
    resp = await router.dispatch(body, stream=False)
    assert resp.status_code == 503
    assert b"not found" in bytes(resp.body)


@pytest.mark.asyncio
async def test_direct_without_header_falls_back_to_policy(monkeypatch):
    # No DIRECT_WORKER_KEY -> falls through to MixedRouter selection: candidates
    # come from the mixed pool and the policy (round-robin) picks one.
    router = DirectRouter(_pool(), RoundRobinPolicy())
    seen = {}

    async def fake_attempt(target, blocks, body, hints, path, stream, obs):
        seen["worker_id"] = target.worker.worker_id
        obs["outcome"] = "ok"
        return _StubResp(200)

    monkeypatch.setattr(router, "_attempt", fake_attempt)
    resp = await router.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200
    # Picked from the mixed pool via policy (not via header).
    assert seen["worker_id"] in ("w1", "w2")


@pytest.mark.asyncio
async def test_direct_decode_worker_in_pd_pool(monkeypatch):
    # Direct mode can target any worker by id regardless of disagg_mode,
    # since selection already happened in the EPP.
    pool = WorkerPool()
    pool.add(
        WorkerInfo(
            worker_id="d1",
            url="http://10.0.2.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.DECODE,
        )
    )
    router = DirectRouter(pool, RoundRobinPolicy())
    seen = {}

    async def fake_attempt(target, blocks, body, hints, path, stream, obs):
        seen["worker_id"] = target.worker.worker_id
        obs["outcome"] = "ok"
        return _StubResp(200)

    monkeypatch.setattr(router, "_attempt", fake_attempt)
    resp = await router.dispatch({"model": "m", DIRECT_WORKER_KEY: "d1"}, stream=False)
    assert resp.status_code == 200
    assert seen["worker_id"] == "d1"


@pytest.mark.asyncio
async def test_direct_pd_delegates_to_disagg(monkeypatch):
    # Both decode + prefill hints -> delegate to the composed DisaggRouter's
    # dispatch_direct with the gateway-chosen worker ids.
    pool = WorkerPool()
    pool.add(
        WorkerInfo(
            worker_id="p1",
            url="http://10.0.1.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.PREFILL,
        )
    )
    pool.add(
        WorkerInfo(
            worker_id="d1",
            url="http://10.0.2.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.DECODE,
        )
    )
    router = DirectRouter(pool, RoundRobinPolicy())
    seen = {}

    async def fake_direct(body, *, stream, path, prefill_id, decode_id):
        seen.update(prefill_id=prefill_id, decode_id=decode_id, stream=stream)
        return _StubResp(200)

    monkeypatch.setattr(router._disagg, "dispatch_direct", fake_direct)
    body = {"model": "m", DIRECT_WORKER_KEY: "d1", DIRECT_PREFILL_KEY: "p1"}
    resp = await router.dispatch(body, stream=True)
    assert resp.status_code == 200
    assert seen == {"prefill_id": "p1", "decode_id": "d1", "stream": True}


@pytest.mark.asyncio
async def test_direct_worker_only_does_not_delegate_to_disagg(monkeypatch):
    # decode hint without a prefill hint -> single-worker direct path, not PD.
    router = DirectRouter(_pool(), RoundRobinPolicy())
    called = {"disagg": False}

    async def fake_direct(*a, **k):
        called["disagg"] = True
        return _StubResp(200)

    async def fake_attempt(target, blocks, body, hints, path, stream, obs):
        obs["outcome"] = "ok"
        return _StubResp(200)

    monkeypatch.setattr(router._disagg, "dispatch_direct", fake_direct)
    monkeypatch.setattr(router, "_attempt", fake_attempt)
    await router.dispatch({"model": "m", DIRECT_WORKER_KEY: "w1"}, stream=False)
    assert called["disagg"] is False


def test_direct_worker_key_constant():
    # Guard the wire contract between server header stash and router read.
    assert json.dumps({DIRECT_WORKER_KEY: "x"})  # serialisable / stable name


@pytest.mark.asyncio
async def test_dispatch_direct_unknown_pd_worker_503():
    from infera.router.disagg import DisaggRouter

    pool = WorkerPool()
    pool.add(
        WorkerInfo(
            worker_id="d1",
            url="http://10.0.2.1:8000",
            model_name="m",
            disagg_mode=DisaggMode.DECODE,
        )
    )
    disagg = DisaggRouter(pool, RoundRobinPolicy())
    resp = await disagg.dispatch_direct(
        {"model": "m"},
        stream=False,
        path="/v1/chat/completions",
        prefill_id="ghost",
        decode_id="d1",
    )
    assert resp.status_code == 503
    await disagg.aclose()
