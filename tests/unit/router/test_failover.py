###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""MixedRouter bounded failover: retry another worker only when a dispatch
fails BEFORE any response data has reached the client."""

from __future__ import annotations

import json

import httpx
import pytest

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.mixed import MixedRouter
from infera.router.policy.target import RouteTarget


class _FakePolicy:
    """Picks the first candidate (the router excludes already-tried workers,
    so this yields a deterministic failover order)."""

    def pick(self, candidates, body):
        return RouteTarget(candidates[0]), []

    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        pass


class _FakePool:
    def __init__(self, workers):
        self._workers = workers

    def list_active(self, model=None, mode=None):
        return list(self._workers)


class _FakeNats:
    """Scripted per-worker reply streams. ``scripts[worker_id]`` is a list of
    ``(kind, status, data)`` events the worker emits."""

    def __init__(self, scripts, admit_allow=None):
        self.scripts = scripts
        self.admit_allow = admit_allow  # None => allow all; else set of worker_ids
        self.streamed: list[str] = []

    async def admit(self, worker_id):
        return self.admit_allow is None or worker_id in self.admit_allow

    async def stream(self, worker_id, payload):
        self.streamed.append(worker_id)
        for ev in self.scripts.get(worker_id, []):
            yield ev


def _w(wid, transport="nats"):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport=transport,
    )


def _router(workers, nats, retries=1):
    r = MixedRouter(
        _FakePool(workers), _FakePolicy(), nats_client=nats, request_max_retries=retries
    )
    return r


async def _drain_stream(resp) -> bytes:
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


# --- streaming: failover before first byte -----------------------------------


@pytest.mark.asyncio
async def test_stream_failover_before_first_byte():
    nats = _FakeNats(
        {
            "w1": [(TYPE_ERROR, 504, b"idle timeout before first token")],
            "w2": [
                (TYPE_DATA, None, b"hello "),
                (TYPE_DATA, None, b"world"),
                (TYPE_DONE, 200, b""),
            ],
        }
    )
    r = _router([_w("w1"), _w("w2")], nats, retries=1)
    resp = await r.dispatch({"model": "m"}, stream=True)
    body = await _drain_stream(resp)
    assert body == b"hello world"
    assert nats.streamed == ["w1", "w2"]  # failed over from w1 to w2
    await r.aclose()


# --- streaming: NO retry once data has been sent ------------------------------


@pytest.mark.asyncio
async def test_stream_no_retry_after_first_byte():
    nats = _FakeNats(
        {
            "w1": [(TYPE_DATA, None, b"partial"), (TYPE_ERROR, None, b"crash mid-stream")],
            "w2": [(TYPE_DATA, None, b"SHOULD-NOT-BE-USED"), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router([_w("w1"), _w("w2")], nats, retries=1)
    resp = await r.dispatch({"model": "m"}, stream=True)
    body = await _drain_stream(resp)
    assert b"partial" in body  # first chunk delivered
    assert b"stream failed mid-stream" in body  # error surfaced inline
    assert b"SHOULD-NOT-BE-USED" not in body
    assert nats.streamed == ["w1"]  # committed to w1, no failover
    await r.aclose()


# --- non-streaming: failover then success -------------------------------------


@pytest.mark.asyncio
async def test_unary_failover():
    ok = json.dumps({"id": "ok"}).encode()
    nats = _FakeNats(
        {
            "w1": [(TYPE_ERROR, 502, b"unreachable")],
            "w2": [(TYPE_DATA, None, ok), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router([_w("w1"), _w("w2")], nats, retries=1)
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body))["id"] == "ok"
    assert nats.streamed == ["w1", "w2"]
    await r.aclose()


# --- admission throttle (429) is a pre-first-byte failure -> failover ---------


@pytest.mark.asyncio
async def test_admit_throttle_failover():
    nats = _FakeNats(
        {"w2": [(TYPE_DATA, None, b"ok"), (TYPE_DONE, 200, b"")]},
        admit_allow={"w2"},  # w1 is over backlog -> refused
    )
    r = _router([_w("w1"), _w("w2")], nats, retries=1)
    resp = await r.dispatch({"model": "m"}, stream=True)
    body = await _drain_stream(resp)
    assert body == b"ok"
    assert nats.streamed == ["w2"]  # w1 refused admission, never streamed
    await r.aclose()


# --- retries exhausted -> return the last error -------------------------------


@pytest.mark.asyncio
async def test_retries_exhausted_returns_error():
    nats = _FakeNats({"w1": [(TYPE_ERROR, 504, b"timeout")]})
    r = _router([_w("w1")], nats, retries=3)  # only 1 worker -> no alternate
    resp = await r.dispatch({"model": "m"}, stream=True)
    assert resp.status_code == 504
    assert nats.streamed == ["w1"]
    await r.aclose()


@pytest.mark.asyncio
async def test_retries_disabled_single_attempt():
    nats = _FakeNats(
        {
            "w1": [(TYPE_ERROR, 502, b"boom")],
            "w2": [(TYPE_DATA, None, b"ok"), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router([_w("w1"), _w("w2")], nats, retries=0)  # no failover
    resp = await r.dispatch({"model": "m"}, stream=True)
    assert resp.status_code == 502
    assert nats.streamed == ["w1"]  # only one attempt
    await r.aclose()


# --- HTTP transport failover (no NATS) ----------------------------------------


@pytest.mark.asyncio
async def test_http_unary_failover():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "w1":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"id": "ok-http"})

    r = _router([_w("w1", transport="http"), _w("w2", transport="http")], nats=None, retries=1)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body))["id"] == "ok-http"
    await r.aclose()
