###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""PD engine abort helpers and client-disconnect abort on the stream path."""

from __future__ import annotations

import asyncio
import json

import anyio
import httpx
import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.pd_abort import (
    abort_engine_request,
    abort_request_ids,
    abort_url,
    prefill_drain_timeout_s,
    rid_for_room,
)
from infera.router.policy.target import RouteTarget
from infera.server.metrics import RequestObserver


def test_rid_for_room_is_stable():
    assert rid_for_room(7) == "infera-7"


def test_abort_url_strips_trailing_slash():
    assert abort_url("http://p:8000/") == "http://p:8000/abort_request"


def test_abort_request_ids_expand_parallel_sampling():
    assert abort_request_ids("infera-7", 1) == ["infera-7"]
    assert abort_request_ids("infera-7", 3) == [
        "infera-7_0",
        "infera-7_1",
        "infera-7_2",
    ]


def test_prefill_drain_timeout_default(monkeypatch):
    monkeypatch.delenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", raising=False)
    assert prefill_drain_timeout_s() == 300.0


def test_prefill_drain_timeout_zero_disables(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    assert prefill_drain_timeout_s() == 0.0


@pytest.mark.asyncio
async def test_abort_engine_request_posts_rid():
    seen = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    await abort_engine_request(client, "http://p:8000", "infera-1")
    await client.aclose()
    assert seen == [("http://p:8000/abort_request", {"rid": "infera-1"})]


@pytest.mark.asyncio
async def test_abort_engine_request_posts_each_parallel_sample():
    seen = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["rid"])
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    await abort_engine_request(client, "http://p:8000", "infera-1", n=3)
    await client.aclose()
    assert seen == ["infera-1_0", "infera-1_1", "infera-1_2"]


class _FakePolicy:
    def __init__(self):
        self.finished = 0

    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        self.finished += 1

    def pick(self, workers, body, role_hint=None):
        return RouteTarget(workers[0]), []


class _FakePool:
    def list_active(self, model=None, mode=None):
        return []


def _w(wid, *, transport="http"):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        disagg_mode=DisaggMode.PREFILL if wid.startswith("p") else DisaggMode.DECODE,
        disagg_meta={
            "protocol": "sglang-bootstrap",
            "params": {"bootstrap_addr": "p1:30001"},
        },
        request_transport=transport,
    )


@pytest.mark.asyncio
async def test_stream_dual_aborts_on_client_cancel(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []

    class _HangDecode:
        status_code = 200

        async def aiter_raw(self):
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            pass

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/abort_request"):
            aborted.append(json.loads(request.content)["rid"])
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"id": "prefill"})

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    r._DECODE_OPEN_MAX_RETRIES = 0

    async def _open(*_a, **_k):
        return _HangDecode()

    r._open_decode_stream = _open  # type: ignore[method-assign]

    async def _consume():
        async for _chunk in r._stream_dual(
            RequestObserver("disagg"),
            RouteTarget(_w("p1")),
            [],
            RouteTarget(_w("d1")),
            [],
            "http://p1/v1/chat/completions",
            "http://d1/v1/chat/completions",
            {"model": "m", "rid": "infera-9"},
            {"model": "m", "rid": "infera-9"},
        ):
            pass

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert aborted.count("infera-9") >= 2, aborted
    assert r.policy.finished == 2
    await r.aclose()


@pytest.mark.asyncio
async def test_stream_dual_cleanup_is_shielded_from_anyio_cancel_scope(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []
    first_chunk = asyncio.Event()
    scope_ready = asyncio.Event()
    scope_holder = {}

    class _HangDecode:
        status_code = 200

        async def aiter_raw(self):
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            await asyncio.sleep(0)

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/abort_request"):
            aborted.append(json.loads(request.content)["rid"])
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"id": "prefill"})

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    r._DECODE_OPEN_MAX_RETRIES = 0

    async def _open(*_a, **_k):
        return _HangDecode()

    r._open_decode_stream = _open  # type: ignore[method-assign]

    async def _consume():
        with anyio.CancelScope() as scope:
            scope_holder["scope"] = scope
            scope_ready.set()
            async for _chunk in r._stream_dual(
                RequestObserver("disagg"),
                RouteTarget(_w("p1")),
                [],
                RouteTarget(_w("d1")),
                [],
                "http://p1/v1/chat/completions",
                "http://d1/v1/chat/completions",
                {"model": "m", "rid": "infera-10"},
                {"model": "m", "rid": "infera-10"},
            ):
                first_chunk.set()

    task = asyncio.create_task(_consume())
    await scope_ready.wait()
    await first_chunk.wait()
    scope_holder["scope"].cancel()
    await task

    assert aborted.count("infera-10") >= 2, aborted
    assert r.policy.finished == 2
    await r.aclose()


@pytest.mark.asyncio
async def test_stream_dual_async_generator_close_aborts_incomplete_pair(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []

    class _HangDecode:
        status_code = 200

        async def aiter_raw(self):
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            pass

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )

    async def _open(*_args, **_kwargs):
        return _HangDecode()

    async def _abort_pair(p, d, rid, n):
        aborted.append((p.worker_id, d.worker_id, rid, n))

    r._open_decode_stream = _open  # type: ignore[method-assign]
    r._abort_pair = _abort_pair  # type: ignore[method-assign]
    stream = r._stream_dual(
        RequestObserver("disagg"),
        RouteTarget(_w("p1")),
        [],
        RouteTarget(_w("d1")),
        [],
        "http://p1/v1/chat/completions",
        "http://d1/v1/chat/completions",
        {"model": "m", "rid": "infera-11"},
        {"model": "m", "rid": "infera-11"},
    )

    await anext(stream)
    await stream.aclose()

    assert aborted == [("p1", "d1", "infera-11", 1)]
    assert r.policy.finished == 2
    await r.aclose()


@pytest.mark.asyncio
async def test_finish_prefill_records_breaker_on_transport_error():
    r = DisaggRouter(_FakePool(), _FakePolicy())

    async def _boom():
        raise httpx.ConnectError("refused")

    task = asyncio.create_task(_boom())
    await r._finish_prefill(
        task, _w("p1"), _w("d1"), "infera-1", 1, abort=False
    )
    assert r.breaker._entries["p1"].consecutive_failures == 1
    await r.aclose()


@pytest.mark.asyncio
async def test_finish_prefill_never_cancels_protocol_without_request_id(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0.01")
    release = asyncio.Event()

    async def _pending():
        await release.wait()

    r = DisaggRouter(_FakePool(), _FakePolicy())
    task = asyncio.create_task(_pending())
    await r._finish_prefill(task, _w("p1"), _w("d1"), None, 1, abort=True)
    assert not task.cancelled()
    assert not task.done()

    await r._finish_prefill(task, _w("p1"), _w("d1"), None, 1, abort=False)
    assert not task.cancelled()
    assert not task.done()
    release.set()
    await task
    await r.aclose()


@pytest.mark.asyncio
async def test_abort_worker_uses_nats_transport_for_nats_worker():
    payloads = []

    class _Nats:
        async def stream(self, worker_id, payload):
            payloads.append((worker_id, payload))
            yield ("done", 200, b"")

    r = DisaggRouter(_FakePool(), _FakePolicy(), nats_client=_Nats())
    await r._abort_worker_request(_w("p1", transport="nats"), "infera-2", 2)

    assert [payload["body"]["rid"] for _, payload in payloads] == [
        "infera-2_0",
        "infera-2_1",
    ]
    assert all(payload["path"] == "/abort_request" for _, payload in payloads)
    await r.aclose()


@pytest.mark.asyncio
async def test_decode_open_does_not_retry_ambiguous_read_error():
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("request may already be accepted", request=request)

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    with pytest.raises(httpx.ReadError):
        await r._open_decode_stream("http://d1/generate", {"rid": "infera-1"})

    assert calls == 1
    await r.aclose()


@pytest.mark.asyncio
async def test_unary_worker_failure_aborts_both_sglang_legs():
    aborted = []

    class _Pool:
        def list_active(self, model=None, mode=None):
            return [_w("p1")] if mode == DisaggMode.PREFILL else [_w("d1")]

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            aborted.append((request.url.host, json.loads(request.content)["rid"]))
            return httpx.Response(200)
        if request.url.host == "p1":
            return httpx.Response(500, json={"error": "KVTransferError"})
        return httpx.Response(200, json={"choices": []})

    r = DisaggRouter(_Pool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    response = await r.dispatch({"model": "m"}, stream=False)

    assert response.status_code == 200
    assert sorted(host for host, _ in aborted) == ["d1", "p1"]
    await r.aclose()
