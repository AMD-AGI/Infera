###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The breaker's PD call sites, exercised on the paths that actually reach them.

These are the streaming generators in ``disagg.py``. They only run when a
decode leg is unreachable -- which is exactly the condition the breaker exists
for, and exactly what no existing test drove. Two of the five record sites were
written against a local name (``d``) that does not exist in these scopes; the
result would have been a ``NameError`` raised *while handling* a decode outage,
turning a clean SSE error into a 500. Lint caught it, but nothing executed it.
"""

from __future__ import annotations

import json

import httpx
import pytest

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.policy.target import RouteTarget
from infera.server.metrics import RequestObserver


class _FakePolicy:
    def pick(self, candidates, body, role_hint=None):
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


def _w(wid):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport="http",
    )


def _router():
    r = DisaggRouter(_FakePool([_w("p1"), _w("d1")]), _FakePolicy())
    # Every send fails at the transport layer: the decode leg is unreachable.
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request))
        )
    )
    # The pre-flight retry loop sleeps between attempts; not worth the wall time.
    r._DECODE_OPEN_MAX_RETRIES = 0
    return r


async def _drain(agen) -> bytes:
    out = b""
    async for chunk in agen:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


def _obs() -> RequestObserver:
    """The SLA observer the streaming generators close on their way out.

    These tests call the generators directly, bypassing the dispatch wrapper
    that normally supplies one. It records nothing here: the observer only
    emits for an outcome of "ok", and every request below fails.
    """
    return RequestObserver("disagg")


@pytest.mark.asyncio
async def test_decode_only_stream_records_failure_on_unreachable():
    r = _router()
    d_target = RouteTarget(_w("d1"))
    body = await _drain(
        r._stream_decode_only(_obs(), d_target, [], "http://d1/v1/chat/completions", {"model": "m"})
    )
    assert b"decode unreachable" in body, "client must get a clean SSE error, not a traceback"
    assert r.breaker.state_of("d1").value == "closed", "one failure is below the threshold"
    for _ in range(2):
        await _drain(
            r._stream_decode_only(
                _obs(), d_target, [], "http://d1/v1/chat/completions", {"model": "m"}
            )
        )
    assert r.breaker.state_of("d1").value == "open", "three unreachable decodes must trip it"
    await r.aclose()


@pytest.mark.asyncio
async def test_dual_stream_records_failure_on_unreachable_decode():
    """Only the decode leg is broken here. Failing both -- which is what a
    transport that refuses everything does -- cannot show that the pools are
    scored independently, because then prefill deserves to trip too.
    """
    r = _router()

    def _only_decode_is_down(request):
        if "d1" in str(request.url):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"id": "x"})

    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_only_decode_is_down))

    p_target = RouteTarget(_w("p1"))
    d_target = RouteTarget(_w("d1"))
    for _ in range(3):
        body = await _drain(
            r._stream_dual(
                _obs(),
                p_target,
                [],
                d_target,
                [],
                "http://p1/v1/chat/completions",
                "http://d1/v1/chat/completions",
                {"model": "m"},
                {"model": "m"},
            )
        )
        assert b"decode unreachable" in body

    assert r.breaker.state_of("d1").value == "open"
    # The prefill leg is a separate pool: a wedged decode must not evict it.
    assert r.breaker.state_of("p1").value == "closed"
    await r.aclose()


class _RolePool:
    """Unlike _FakePool, hands back the pool the caller actually asked for, so
    a dispatch gets a real prefill/decode pair rather than the same worker twice."""

    def __init__(self, prefill, decode):
        self._by_mode = {DisaggMode.PREFILL: [prefill], DisaggMode.DECODE: [decode]}

    def list_active(self, model=None, mode=None):
        return list(self._by_mode.get(mode, []))


def _pd_worker(wid, mode, transport="http"):
    meta = {"protocol": "sglang-bootstrap"}
    if mode is DisaggMode.PREFILL:
        meta["params"] = {"bootstrap_addr": f"{wid}:9000"}
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport=transport,
        disagg_mode=mode,
        disagg_meta=meta,
    )


def _nats_router(*, fail_decode, prefill_status=200):
    """A PD pair that both registered for the NATS transport, which is what
    selects the NATS dispatch path."""
    return DisaggRouter(
        _RolePool(
            _pd_worker("p1", DisaggMode.PREFILL, transport="nats"),
            _pd_worker("d1", DisaggMode.DECODE, transport="nats"),
        ),
        _FakePolicy(),
        nats_client=_FakeNatsPD(fail_decode=fail_decode, prefill_status=prefill_status),
    )


def _ok_router():
    """A PD router whose every leg answers 200."""
    r = DisaggRouter(
        _RolePool(_pd_worker("p1", DisaggMode.PREFILL), _pd_worker("d1", DisaggMode.DECODE)),
        _FakePolicy(),
    )
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"id": "x", "choices": [{"message": {"content": "hi"}}]}
            )
        )
    )
    return r


@pytest.mark.asyncio
async def test_a_served_request_clears_the_failure_count():
    """Without a success recorded anywhere, "three consecutive failures" decays
    into "three failures ever": the counter only climbs, so a healthy worker
    that fails once a day trips on day three."""
    r = _ok_router()
    for _ in range(2):
        r.breaker.record_failure("d1")
    assert r.breaker.snapshot()["d1"]["consecutive_failures"] == 2

    await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.snapshot()["d1"]["consecutive_failures"] == 0
    assert r.breaker.state_of("d1").value == "closed"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_healthy_leg_does_not_launder_a_broken_one():
    """The two legs are two workers whose health is independent, so each is
    scored from its own response.

    Scoring both off the single client-facing status code let the decode leg's
    200 count as evidence for a prefill that had just 500'd -- resetting its
    failure count, and reopening a breaker that was already open.
    """

    def _prefill_is_broken(request):
        if "p1" in str(request.url):
            return httpx.Response(500, json={"error": "prefill exploded"})
        return httpx.Response(200, json={"id": "x", "choices": [{"message": {"content": "hi"}}]})

    r = DisaggRouter(
        _RolePool(_pd_worker("p1", DisaggMode.PREFILL), _pd_worker("d1", DisaggMode.DECODE)),
        _FakePolicy(),
    )
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_prefill_is_broken))

    for _ in range(3):
        resp = await r.dispatch({"model": "m"}, stream=False)
        assert resp.status_code == 200, "decode answers, so the client still gets 200"

    assert r.breaker.state_of("p1").value == "open", (
        "a prefill that 500s every request must trip, even though the decode leg beside it succeeds"
    )
    assert r.breaker.state_of("d1").value == "closed", "the healthy leg is untouched"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_streaming_dispatch_is_not_scored_before_it_runs():
    """A StreamingResponse is returned before its generator is touched: the
    decode leg has not been POSTed and its 200 is Starlette's default, not an
    outcome. Scoring it there records success for both roles before either was
    dispatched and resets the count the legs are about to raise -- a decode
    worker failing every request would never trip.

    Drives dispatch(stream=True), which is the production path; the older tests
    call the generators directly and so cannot see this.
    """
    r = DisaggRouter(
        _RolePool(_pd_worker("p1", DisaggMode.PREFILL), _pd_worker("d1", DisaggMode.DECODE)),
        _FakePolicy(),
    )
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused", request=request))
        )
    )
    r._DECODE_OPEN_MAX_RETRIES = 0

    for i in range(3):
        resp = await r.dispatch({"model": "m"}, stream=True)
        assert await _drain(resp.body_iterator), f"request {i} produced no body"

    assert r.breaker.state_of("d1").value == "open", (
        "three unreachable decodes must trip the breaker; if this is closed the "
        "wrapper scored the stream before the decode leg ran"
    )
    await r.aclose()


@pytest.mark.asyncio
async def test_a_tripped_pd_worker_recovers_after_a_good_probe():
    """The probe is dispatched and succeeds; if nothing records that, the
    worker stays half-open with its probe slot held and never routes again."""
    r = _ok_router()
    for _ in range(3):
        r.breaker.record_failure("d1")
    assert r.breaker.state_of("d1").value == "open"

    # Let the cooldown lapse so the next dispatch is the half-open probe.
    r.breaker._entries["d1"].opens_until = 0.0

    await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("d1").value == "closed", "a good probe must close it"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_decode_outage_does_not_trip_the_prefill_worker():
    """asyncio.gather raises whichever leg failed, without saying which. Blaming
    a fixed one means a decode that refuses connections evicts the healthy
    prefill worker from rotation while the broken decode is never scored at
    all -- the exact inversion of what the breaker is for."""

    def _only_decode_is_down(request):
        if "d1" in str(request.url):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"id": "x"})

    r = DisaggRouter(
        _RolePool(_pd_worker("p1", DisaggMode.PREFILL), _pd_worker("d1", DisaggMode.DECODE)),
        _FakePolicy(),
    )
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_only_decode_is_down))

    for _ in range(3):
        await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("d1").value == "open", "the leg that refused must be the one scored"
    assert r.breaker.state_of("p1").value == "closed", "the healthy leg must not be evicted"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_stream_that_dies_after_its_headers_is_not_a_success():
    """A 200 header only means the request was accepted. Treating it as
    recovery resets the failure count, so a decode worker that answers 200 and
    then sends nothing -- which is precisely the "healthy to the platform,
    broken for inference" profile the breaker exists for -- can never trip, and
    erases real failures on its way."""
    r = DisaggRouter(
        _RolePool(_pd_worker("p1", DisaggMode.PREFILL), _pd_worker("d1", DisaggMode.DECODE)),
        _FakePolicy(),
    )

    def _headers_then_nothing(request):
        if "d1" in str(request.url):
            # 200, then the body raises as soon as it is read.
            return httpx.Response(200, stream=_DyingStream())
        return httpx.Response(200, json={"id": "x"})

    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_headers_then_nothing))

    for _ in range(2):
        r.breaker.record_failure("d1")
    before = r.breaker.snapshot()["d1"]["consecutive_failures"]
    assert before == 2

    resp = await r.dispatch({"model": "m"}, stream=True)
    await _drain(resp.body_iterator)

    after = r.breaker.snapshot()["d1"]["consecutive_failures"]
    assert after >= before, (
        f"consecutive_failures went {before} -> {after}: a stream that produced "
        "no output was scored as evidence of health"
    )
    await r.aclose()


class _DyingStream(httpx.AsyncByteStream):
    """Headers arrive, then the body fails -- no bytes ever reach the client."""

    async def __aiter__(self):
        raise httpx.ReadError("connection died after headers")
        yield b""  # unreachable; makes this an async generator


@pytest.mark.asyncio
async def test_the_nats_transport_scores_its_legs_too():
    """The NATS paths do not use the HTTP client, so scoring placed at HTTP
    response sites misses them entirely. A decode worker failing every request
    over NATS would be invisible to the breaker, and -- worse -- one already
    open could never be closed again, since nothing would ever record the
    success that ends its half-open state."""
    r = _nats_router(fail_decode=True)

    for _ in range(3):
        await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("d1").value == "open", "a failing NATS decode leg must trip"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_prefill_that_failed_over_nats_is_not_recorded_as_healthy():
    """A `done` frame says the request finished, not that it succeeded.

    The worker proxies whatever its engine returned, so a 500 comes back as
    `done` with rs-status 500 -- an `error` frame means the transport failed,
    which is a different thing. Scoring on the frame kind alone therefore reads
    every failed prefill as evidence of health, and this is the one leg where
    that is invisible from anywhere else: its reply is discarded, so nothing
    else observes it, and a prefill that never registers its bootstrap_room
    leaves every decode paired with it hanging on KVPoll. The HTTP path already
    scores the same status correctly.
    """
    r = _nats_router(fail_decode=False, prefill_status=500)

    for _ in range(3):
        await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("p1").value == "open", (
        "a prefill leg failing every request must trip its breaker"
    )
    await r.aclose()


@pytest.mark.asyncio
async def test_a_prefill_4xx_over_nats_is_not_held_against_the_worker():
    """A 4xx is the request's fault -- every worker would answer the same, so
    it must not accumulate towards tripping."""
    r = _nats_router(fail_decode=False, prefill_status=400)

    for _ in range(5):
        await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("p1").value == "closed"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_good_nats_probe_closes_the_breaker():
    r = _nats_router(fail_decode=False)
    for _ in range(3):
        r.breaker.record_failure("d1")
    r.breaker._entries["d1"].opens_until = 0.0

    await r.dispatch({"model": "m"}, stream=False)

    assert r.breaker.state_of("d1").value == "closed", (
        "a NATS worker that answers cleanly must be able to recover; otherwise "
        "it stays half-open forever, throttled to one request per probe window"
    )
    await r.aclose()


class _FakeNatsPD:
    """Scripted NATS transport: decode either answers or errors, prefill is fine.

    ``prefill_status`` models a prefill whose *engine* failed. That is not an
    error frame: the worker proxies whatever the engine said, so a 500 arrives
    as ``done`` with ``rs-status: 500`` exactly as a 200 would.
    """

    def __init__(self, *, fail_decode: bool, prefill_status: int = 200):
        self.fail_decode = fail_decode
        self.prefill_status = prefill_status

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        if worker_id == "d1" and self.fail_decode:
            yield (TYPE_ERROR, 502, b"decode exploded")
            return
        if worker_id == "p1" and self.prefill_status != 200:
            yield (TYPE_DATA, None, b'{"error":"prefill exploded"}')
            yield (TYPE_DONE, self.prefill_status, b"")
            return
        yield (TYPE_DATA, None, json.dumps({"id": "x"}).encode())
        yield (TYPE_DONE, 200, b"")
