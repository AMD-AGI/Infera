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

import httpx
import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.policy.target import RouteTarget


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


@pytest.mark.asyncio
async def test_decode_only_stream_records_failure_on_unreachable():
    r = _router()
    d_target = RouteTarget(_w("d1"))
    body = await _drain(
        r._stream_decode_only(d_target, [], "http://d1/v1/chat/completions", {"model": "m"})
    )
    assert b"decode unreachable" in body, "client must get a clean SSE error, not a traceback"
    assert r.breaker.state_of("d1").value == "closed", "one failure is below the threshold"
    for _ in range(2):
        await _drain(
            r._stream_decode_only(d_target, [], "http://d1/v1/chat/completions", {"model": "m"})
        )
    assert r.breaker.state_of("d1").value == "open", "three unreachable decodes must trip it"
    await r.aclose()


@pytest.mark.asyncio
async def test_dual_stream_records_failure_on_unreachable_decode():
    r = _router()
    p_target = RouteTarget(_w("p1"))
    d_target = RouteTarget(_w("d1"))
    for _ in range(3):
        body = await _drain(
            r._stream_dual(
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


def _pd_worker(wid, mode):
    meta = {"protocol": "sglang-bootstrap"}
    if mode is DisaggMode.PREFILL:
        meta["params"] = {"bootstrap_addr": f"{wid}:9000"}
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport="http",
        disagg_mode=mode,
        disagg_meta=meta,
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
