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
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
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
    assert b"failed mid-stream" in body  # error surfaced inline
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


# --- circuit breaker: failure memory ACROSS requests (issue #82) ---------------


@pytest.mark.asyncio
async def test_breaker_stops_reselecting_a_dead_worker():
    """The regression behind issue #82, through the real MixedRouter.

    Failover already made all ten of these requests succeed, so a test that
    only checked status codes passed both before and after the fix. What was
    broken is the cost: ``tried`` is per-request, so a worker that is broken
    for inference but healthy to discovery was re-picked on *every* request and
    every one of them paid a wasted round trip. ``nats.streamed`` is the
    assertion that matters.
    """
    scripts = {
        "w1": [(TYPE_ERROR, 502, b"wedged")],
        "w2": [(TYPE_DATA, None, b"ok"), (TYPE_DONE, 200, b"")],
    }
    nats = _FakeNats(scripts)
    r = _router([_w("w1"), _w("w2")], nats, retries=1)
    for _ in range(10):
        resp = await r.dispatch({"model": "m"}, stream=True)
        assert await _drain_stream(resp) == b"ok", "failover must still serve every request"

    # _FakePolicy always picks candidates[0], so w1 is offered every request
    # until the breaker (threshold 3) takes it out; its 5s cooldown does not
    # elapse during the test.
    assert nats.streamed.count("w1") == 3, (
        f"w1 dispatched {nats.streamed.count('w1')} times; expected 3 "
        "(it was 10 before the breaker existed)"
    )
    assert nats.streamed.count("w2") == 10
    await r.aclose()


@pytest.mark.asyncio
async def test_breaker_ignores_client_errors():
    """A 400 is the request's fault, and every worker would return it. Counting
    it would circuit-break an entirely healthy fleet on one bad client."""
    nats = _FakeNats({"w1": [(TYPE_ERROR, 400, b"bad request")]})
    r = _router([_w("w1")], nats, retries=0)
    for _ in range(10):
        await r.dispatch({"model": "m"}, stream=True)
    assert nats.streamed.count("w1") == 10, "4xx must not take a healthy worker out of rotation"
    await r.aclose()


@pytest.mark.asyncio
async def test_breaker_recovers_after_cooldown():
    """A worker that comes back must be picked up again, not stay excluded."""
    scripts = {
        "w1": [(TYPE_ERROR, 502, b"wedged")],
        "w2": [(TYPE_DATA, None, b"ok"), (TYPE_DONE, 200, b"")],
    }
    nats = _FakeNats(scripts)
    r = _router([_w("w1"), _w("w2")], nats, retries=1)

    clock = [1000.0]
    r.breaker.now = lambda: clock[0]

    for _ in range(3):
        await _drain_stream(await r.dispatch({"model": "m"}, stream=True))
    assert nats.streamed.count("w1") == 3  # tripped

    await _drain_stream(await r.dispatch({"model": "m"}, stream=True))
    assert nats.streamed.count("w1") == 3, "still open during the cooldown"

    scripts["w1"] = [(TYPE_DATA, None, b"back"), (TYPE_DONE, 200, b"")]
    clock[0] += 5.1
    body = await _drain_stream(await r.dispatch({"model": "m"}, stream=True))
    assert body == b"back", "the half-open probe must reach the recovered worker"
    assert r.breaker.state_of("w1").value == "closed"
    await r.aclose()


# --- unary 5xx must fail over too (the gap the breaker fell through) ---------


@pytest.mark.asyncio
async def test_unary_http_fails_over_on_5xx():
    """A worker 500 before any byte reached the client is exactly what failover
    is for. This path used to return it verbatim, so a non-streaming request
    over HTTP -- the default in every k8s example -- never failed over and never
    reached the circuit breaker."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "w1":
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"id": "ok"})

    r = _router([_w("w1", transport="http"), _w("w2", transport="http")], nats=None, retries=1)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body))["id"] == "ok"
    await r.aclose()


@pytest.mark.asyncio
async def test_unary_http_does_not_fail_over_on_4xx():
    """The request is bad, not the worker. Every worker would answer the same,
    so retrying only triples the latency of an error the client must see."""
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        return httpx.Response(400, json={"error": "bad request"})

    r = _router([_w("w1", transport="http"), _w("w2", transport="http")], nats=None, retries=1)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 400
    assert hits == ["w1"], f"4xx must not be retried, but hit {hits}"
    await r.aclose()


@pytest.mark.asyncio
async def test_unary_5xx_trips_the_breaker():
    """The consequence that made this worth fixing: without failover the
    breaker never saw a unary failure, so a wedged worker was re-picked
    forever on the most common configuration."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad":
            return httpx.Response(503, json={"error": "wedged"})
        return httpx.Response(200, json={"id": "ok"})

    r = _router([_w("bad", transport="http"), _w("good", transport="http")], nats=None, retries=1)
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    for _ in range(3):
        assert (await r.dispatch({"model": "m"}, stream=False)).status_code == 200
    assert r.breaker.state_of("bad").value == "open"
    await r.aclose()


# --- half a PD deployment must say so --------------------------------------


class _ModePool:
    """Pool that can answer per-disagg-mode, unlike _FakePool."""

    def __init__(self, workers):
        self._w = workers

    def list_active(self, model=None, mode=None):
        return [w for w in self._w if mode is None or w.disagg_mode == mode]


def _pd_worker(wid, mode):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        disagg_mode=mode,
        request_transport="http",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "present,missing",
    [(DisaggMode.PREFILL, "decode"), (DisaggMode.DECODE, "prefill")],
)
async def test_half_a_pd_deployment_names_the_empty_pool(present, missing):
    """Scaling either PD pool to zero fails closed -- correctly -- but used to
    report "no active mixed worker", which points at something the operator
    never deployed while the surviving pool sits right there."""
    from infera.router.auto import AutoRouter

    r = AutoRouter(_ModePool([_pd_worker("w1", present)]), _FakePolicy())
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 503
    body = json.loads(bytes(resp.body))["error"]
    assert missing in body and "PD dispatch requires both pools" in body, body
    assert "mixed" not in body, f"must not blame mixed workers: {body}"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_mixed_worker_still_absorbs_a_half_pd_fleet():
    """A mixed worker alongside half a PD pool can serve, so the 503 must not
    fire -- this is the rolling-upgrade case."""
    from infera.router.auto import AutoRouter

    pool = _ModePool([_pd_worker("p", DisaggMode.PREFILL), _pd_worker("m1", DisaggMode.MIXED)])
    r = AutoRouter(pool, _FakePolicy(), nats_client=None, request_max_retries=0)
    r._mixed._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"id": "ok"}))
    )
    resp = await r.dispatch({"model": "m"}, stream=False)
    assert resp.status_code == 200
    await r.aclose()
