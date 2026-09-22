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

from infera.common.nats_request import TYPE_DATA, TYPE_DONE
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.cache_control import parse_cache_hints
from infera.router.disagg import DisaggRouter
from infera.router.pd_abort import (
    ABORT_PATH,
    abort_engine_request,
    abort_request_ids,
    abort_url,
    prefill_drain_timeout_s,
    rid_for_room,
)
from infera.router.policy.target import RouteTarget
from infera.server import metrics
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
async def test_finish_prefill_drain_timeout_cancels_task_without_request_id(monkeypatch):
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0.01")
    release = asyncio.Event()

    async def _pending():
        await release.wait()

    r = DisaggRouter(_FakePool(), _FakePolicy())
    task = asyncio.create_task(_pending())

    await r._finish_prefill(task, _w("p1"), _w("d1"), None, 1, abort=False)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    await r.aclose()


@pytest.mark.asyncio
async def test_finish_prefill_abort_cancels_task_without_request_id():
    """The vLLM and ATOM connectors have no remote abort endpoint, so closing
    the prefill POST is the only way to drop the request there: an abort must
    cancel the task even when the protocol forged no rid to abort by."""
    aborted = []
    release = asyncio.Event()

    async def _pending():
        await release.wait()

    r = DisaggRouter(_FakePool(), _FakePolicy())

    async def _abort_pair(*args):
        aborted.append(args)

    r._abort_pair = _abort_pair  # type: ignore[method-assign]
    task = asyncio.create_task(_pending())

    await r._finish_prefill(task, _w("p1"), _w("d1"), None, 1, abort=True)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert task.cancelled()
    assert aborted == [], "no rid means nothing to abort by request id"
    release.set()
    await r.aclose()


@pytest.mark.asyncio
async def test_finish_prefill_cancel_during_drain_does_not_abort():
    """abort=False means the decode stream already finished: a cancel of the
    waiter must not POST /abort_request or cancel the shielded prefill task."""
    aborted = []
    release = asyncio.Event()

    async def _pending():
        await release.wait()

    r = DisaggRouter(_FakePool(), _FakePolicy())

    async def _abort_pair(*args):
        aborted.append(args)

    r._abort_pair = _abort_pair  # type: ignore[method-assign]
    task = asyncio.create_task(_pending())

    async def _drain():
        await r._finish_prefill(task, _w("p1"), _w("d1"), "infera-21", 1, abort=False)

    drain = asyncio.create_task(_drain())
    await asyncio.sleep(0)
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    assert aborted == []
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


@pytest.mark.asyncio
async def test_unary_client_cancel_aborts_both_legs():
    """A client that drops mid-dispatch leaves both engines holding the
    request: the POSTs are cancelled, but neither engine hears about it, so
    the prefill slot stays busy until the transfer timeout."""
    aborted = []
    dispatched = asyncio.Event()

    class _Pool:
        def list_active(self, model=None, mode=None):
            return [_w("p1")] if mode == DisaggMode.PREFILL else [_w("d1")]

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort_request":
            aborted.append((request.url.host, json.loads(request.content)["rid"]))
            return httpx.Response(200, json={})
        dispatched.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json={})

    r = DisaggRouter(_Pool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    task = asyncio.create_task(r.dispatch({"model": "m", "n": 3}, stream=False))
    await dispatched.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted({host for host, _ in aborted}) == ["d1", "p1"], aborted
    assert sorted({rid.rsplit("_", 1)[1] for _, rid in aborted}) == ["0", "1", "2"], aborted
    assert r.policy.finished == 2
    await r.aclose()


@pytest.mark.asyncio
async def test_prefill_drain_timeout_scores_the_prefill_worker(monkeypatch):
    """A drain that never lands is the wedged-prefill signal the breaker is
    for, with or without a request id to abort by."""
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0.01")
    reason = "prefill_drain_timeout"
    before = metrics.pd_bootstrap_failures_total.labels(reason=reason)._value.get()

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    hung = [asyncio.create_task(asyncio.Event().wait()) for _ in range(2)]

    await r._finish_prefill(hung[0], _w("p1"), _w("d1"), "infera-12", 1, abort=False)
    assert r.breaker._entries["p1"].consecutive_failures == 1

    await r._finish_prefill(hung[1], _w("p1"), _w("d1"), None, 1, abort=False)
    assert r.breaker._entries["p1"].consecutive_failures == 2, (
        "a wedged drain is the worker's fault even when the protocol has no rid"
    )

    after = metrics.pd_bootstrap_failures_total.labels(reason=reason)._value.get()
    assert after - before == 2

    for task in hung:
        task.cancel()
    await asyncio.gather(*hung, return_exceptions=True)
    await r.aclose()


class _ScriptedNats:
    """NATS transport replaying one decode chunk then a done frame."""

    def __init__(self, status: int):
        self._status = status

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        yield (TYPE_DATA, None, b'data: {"id":"x"}\n\n')
        yield (TYPE_DONE, self._status, b"")


async def _drain_nats_stream(r, rid="infera-13", n=2):
    """Run _stream_dual_nats against a finished prefill task."""
    p_task = asyncio.create_task(asyncio.sleep(0))
    chunks = []
    async for chunk in r._stream_dual_nats(
        RequestObserver("disagg"),
        RouteTarget(_w("p1", transport="nats")),
        [],
        RouteTarget(_w("d1", transport="nats")),
        [],
        {"path": "/v1/chat/completions", "stream": True, "headers": None, "body": {}},
        p_task,
        rid=rid,
        n=n,
    ):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_nats_stream_done_5xx_aborts_the_pair(monkeypatch):
    """A done frame reports that the request finished, not that it succeeded:
    a 5xx there is a failed decode whose pair still holds engine slots."""
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []

    r = DisaggRouter(_FakePool(), _FakePolicy(), nats_client=_ScriptedNats(500))

    async def _abort_pair(p, d, rid, n):
        aborted.append((p.worker_id, d.worker_id, rid, n))

    r._abort_pair = _abort_pair  # type: ignore[method-assign]

    assert await _drain_nats_stream(r)

    assert aborted == [("p1", "d1", "infera-13", 2)]
    assert r.breaker._entries["d1"].consecutive_failures == 1
    await r.aclose()


@pytest.mark.asyncio
async def test_nats_stream_done_4xx_does_not_abort_the_pair(monkeypatch):
    """A 4xx is the request's fault, and the unary path does not abort on it."""
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []

    r = DisaggRouter(_FakePool(), _FakePolicy(), nats_client=_ScriptedNats(400))

    async def _abort_pair(p, d, rid, n):
        aborted.append((p.worker_id, d.worker_id, rid, n))

    r._abort_pair = _abort_pair  # type: ignore[method-assign]

    assert await _drain_nats_stream(r)

    assert aborted == []
    assert r.breaker.snapshot().get("d1", {}).get("consecutive_failures", 0) == 0
    await r.aclose()


class _ScriptedDecode:
    """Decode leg replaying raw SSE chunks, then EOF."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        pass


class _RolePool:
    """Hands back the pool the caller asked for, so a dispatch gets a pair."""

    def __init__(self, transport="http"):
        self._transport = transport

    def list_active(self, model=None, mode=None):
        wid = "p1" if mode == DisaggMode.PREFILL else "d1"
        return [_w(wid, transport=self._transport)]


@pytest.mark.asyncio
async def test_responses_stream_completion_event_is_not_aborted(monkeypatch):
    """SGLang ends a /v1/responses stream with `event: response.completed`,
    never `data: [DONE]`, and the raw chunking can split it. Matching only the
    chat sentinel reads a finished Responses stream as truncated and aborts a
    pair that already answered."""
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "0")
    aborted = []

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )

    async def _open(*_a, **_k):
        return _ScriptedDecode(
            [
                b"event: response.in_progress\ndata: {}\n\n",
                b"event: response.comp",
                b'leted\ndata: {"id":"resp_1"}\n\n',
            ]
        )

    async def _abort_pair(p, d, rid, n):
        aborted.append((rid, n))

    r._open_decode_stream = _open  # type: ignore[method-assign]
    r._abort_pair = _abort_pair  # type: ignore[method-assign]

    chunks = [
        chunk
        async for chunk in r._stream_dual(
            RequestObserver("disagg"),
            RouteTarget(_w("p1")),
            [],
            RouteTarget(_w("d1")),
            [],
            "http://p1/v1/responses",
            "http://d1/v1/responses",
            {"model": "m", "rid": "infera-20"},
            {"model": "m", "rid": "infera-20"},
            path="/v1/responses",
        )
    ]

    assert len(chunks) == 3
    assert aborted == []
    assert r.policy.finished == 2
    await r.aclose()


@pytest.mark.asyncio
async def test_responses_bodies_carry_request_id_for_sglang():
    """SGLang's ResponsesRequest drops an unknown `rid`, but keeps
    `request_id`, which serving_responses passes to the scheduler as the id
    /abort_request matches. Without it the forged rid aborts nothing."""
    bodies = []

    def _handler(request: httpx.Request) -> httpx.Response:
        bodies.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"id": "x"})

    r = DisaggRouter(_RolePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    await r.dispatch({"model": "m"}, stream=False, path="/v1/responses")
    await r.dispatch({"model": "m"}, stream=False, path="/v1/chat/completions")

    responses_bodies = [body for path, body in bodies if path == "/v1/responses"]
    assert len(responses_bodies) == 2, "both legs must be annotated"
    assert all(body["request_id"] == body["rid"] for body in responses_bodies)
    chat_bodies = [body for path, body in bodies if path == "/v1/chat/completions"]
    assert chat_bodies and all("request_id" not in body for body in chat_bodies)
    await r.aclose()


class _HangingNats:
    """Decode never answers; abort requests are recorded and acknowledged."""

    def __init__(self):
        self.aborted = []
        self.decode_open = asyncio.Event()

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        if payload["path"] == ABORT_PATH:
            self.aborted.append((worker_id, payload["body"]["rid"]))
            yield (TYPE_DONE, 200, b"")
            return
        if worker_id == "d1":
            self.decode_open.set()
        await asyncio.Event().wait()
        yield (TYPE_DONE, 200, b"")  # unreachable


@pytest.mark.asyncio
async def test_nats_unary_client_cancel_aborts_both_legs():
    """The cancellation that drops the client also cancels the cleanup awaits,
    so the abort has to run shielded; otherwise both NATS legs stay inflight
    holding engine slots."""
    nats = _HangingNats()
    r = DisaggRouter(_RolePool(transport="nats"), _FakePolicy(), nats_client=nats)
    holder = {}

    async def _run():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            await r.dispatch({"model": "m", "n": 2}, stream=False)

    task = asyncio.create_task(_run())
    await nats.decode_open.wait()
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)
    await asyncio.sleep(0)

    assert sorted(worker_id for worker_id, _ in nats.aborted) == ["d1", "d1", "p1", "p1"]
    assert sorted(rid.rsplit("_", 1)[1] for _, rid in nats.aborted) == ["0", "0", "1", "1"]
    assert r.policy.finished == 2
    assert not r._pending_prefill_tasks, "the prefill drain must not outlive the cancel"
    await r.aclose()


@pytest.mark.asyncio
async def test_completed_stream_cancel_does_not_wait_for_prefill_drain(monkeypatch):
    """A cancel arriving after the terminal marker must release the policy
    slots immediately: the client already has the full response, so waiting out
    the prefill drain inside the shield only pins capacity."""
    monkeypatch.setenv("INFERA_PD_PREFILL_DRAIN_TIMEOUT", "300")
    holder = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == ABORT_PATH:
            return httpx.Response(200, json={})
        await asyncio.Event().wait()  # prefill never lands
        return httpx.Response(200, json={})

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    class _DoneThenIdle(_ScriptedDecode):
        """Terminal marker delivered, then the socket sits idle -- where a
        client that got its answer and went away leaves the generator."""

        async def aiter_raw(self):
            yield b'data: {"id":"x"}\n\n'
            yield b"data: [DONE]\n\n"
            await asyncio.Event().wait()

    async def _open(*_a, **_k):
        return _DoneThenIdle([])

    r._open_decode_stream = _open  # type: ignore[method-assign]
    aborted = []

    async def _abort_pair(*args):
        aborted.append(args)

    r._abort_pair = _abort_pair  # type: ignore[method-assign]

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for _chunk in r._stream_dual(
                RequestObserver("disagg"),
                RouteTarget(_w("p1")),
                [],
                RouteTarget(_w("d1")),
                [],
                "http://p1/v1/chat/completions",
                "http://d1/v1/chat/completions",
                {"model": "m", "rid": "infera-21"},
                {"model": "m", "rid": "infera-21"},
            ):
                pass

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    holder["scope"].cancel()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if r.policy.finished == 2:
            break

    assert r.policy.finished == 2, (
        "the prefill is still hanging, so the slots were held for the drain timeout"
    )
    assert aborted == []
    await asyncio.wait_for(task, timeout=5)
    for pending in list(r._pending_prefill_tasks):
        pending.cancel()
    await r.aclose()


class _SerialProto:
    """Minimal serial-pull protocol: bodies pass through, handoff is empty."""

    name = "vllm-mori-read"
    topology = "serial-pull"

    def annotate_prefill(self, base, p, d, room_id):
        return dict(base)

    def annotate_decode(self, base, p, d, room_id, handoff):
        return dict(base)

    def extract_handoff(self, payload):
        return {}


async def _run_serial(r, *, stream=False):
    """Drive _dispatch_serial with a pass-through serial-pull protocol."""
    return await r._dispatch_serial(
        RequestObserver("disagg"),
        _SerialProto(),
        {"model": "m"},
        parse_cache_hints({}),
        RouteTarget(_w("p1")),
        [],
        "http://p1/v1/chat/completions",
        RouteTarget(_w("d1")),
        [],
        "http://d1/v1/chat/completions",
        7,
        stream,
        None,
    )


@pytest.mark.asyncio
async def test_serial_prefill_cancel_releases_both_policy_slots():
    """Serial-pull holds the prefill and decode slots until the prefill
    response lands. A client drop during that POST must release both exactly
    once; the cancelled request closes the prefill connection, which is what
    drops the request on connectors without an abort endpoint."""
    paths = []
    prefill_open = asyncio.Event()

    async def _handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        prefill_open.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json={})

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    holder = {}

    async def _run():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            await _run_serial(r)

    task = asyncio.create_task(_run())
    await prefill_open.wait()
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    assert r.policy.finished == 2, "both legs were accounted for before the POST"
    assert paths == ["/v1/chat/completions"], "no abort endpoint on this protocol"
    await r.aclose()


@pytest.mark.asyncio
async def test_decode_only_stream_cancel_closes_the_decode_response():
    """The cancellation that drops the client also cancels the cleanup await,
    so aclose() has to run shielded: otherwise the decode connection stays
    open and the engine keeps the request inflight."""
    closed = []

    class _HangingDecode:
        status_code = 200

        async def aiter_raw(self):
            yield b'data: {"id":"x"}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            # Closing a real connection awaits I/O, so it is a cancellation
            # point: an unshielded teardown never gets this far.
            await asyncio.sleep(0.01)
            closed.append(True)

    r = DisaggRouter(_FakePool(), _FakePolicy())

    async def _open(*_a, **_k):
        return _HangingDecode()

    r._open_decode_stream = _open  # type: ignore[method-assign]
    holder = {}

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for _chunk in r._stream_decode_only(
                RequestObserver("disagg"),
                RouteTarget(_w("d1")),
                [],
                "http://d1/v1/chat/completions",
                {"model": "m"},
            ):
                pass

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    assert closed == [True]
    assert r.policy.finished == 1
    await r.aclose()


class _ConcurrentProto:
    """Minimal sglang-bootstrap protocol: rid only, no bootstrap rewriting."""

    name = "sglang-bootstrap"
    topology = "concurrent"

    def annotate_prefill(self, base, p, d, room_id):
        return {**base, "rid": rid_for_room(room_id)}

    def annotate_decode(self, base, p, d, room_id, handoff):
        return self.annotate_prefill(base, p, d, room_id)

    def extract_handoff(self, payload):
        return {}


def _header_capture_router():
    """Router whose HTTP client records the headers each leg was POSTed with."""
    seen: dict[str, dict[str, str]] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = dict(request.headers)
        return httpx.Response(200, json={})

    r = DisaggRouter(_FakePool(), _FakePolicy())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return r, seen


@pytest.mark.asyncio
async def test_concurrent_legs_carry_the_sglang_rid_override_header():
    """SGLang adopts the forged rid from ``x-override-rid``; ``X-Request-Id``
    is ignored there, so an abort by that id would never match."""
    r, seen = _header_capture_router()
    resp = await r._dispatch_concurrent(
        RequestObserver("disagg"),
        _ConcurrentProto(),
        {"model": "m"},
        parse_cache_hints({}),
        RouteTarget(_w("p1")),
        [],
        "http://p1/v1/chat/completions",
        RouteTarget(_w("d1")),
        [],
        "http://d1/v1/chat/completions",
        7,
        False,
        "infera-7",
    )

    assert resp.status_code == 200
    for leg in ("p1", "d1"):
        assert seen[leg]["x-override-rid"] == "infera-7"
        assert "x-request-id" not in seen[leg]
    await r.aclose()


@pytest.mark.asyncio
async def test_serial_legs_keep_the_generic_request_id_header():
    """Non-SGLang connectors read the forged id from ``X-Request-Id``."""
    r, seen = _header_capture_router()
    await r._dispatch_serial(
        RequestObserver("disagg"),
        _SerialProto(),
        {"model": "m"},
        parse_cache_hints({}),
        RouteTarget(_w("p1")),
        [],
        "http://p1/v1/chat/completions",
        RouteTarget(_w("d1")),
        [],
        "http://d1/v1/chat/completions",
        7,
        False,
        "infera-7",
    )

    for leg in ("p1", "d1"):
        assert seen[leg]["x-request-id"] == "infera-7"
        assert "x-override-rid" not in seen[leg]
    await r.aclose()


class _MixedTransportPool:
    """Prefill registered for NATS, decode for HTTP."""

    def __init__(self):
        self._p = _w("p1", transport="nats")
        self._d = _w("d1", transport="http")

    def list_active(self, model=None, mode=None):
        return [self._p if mode == DisaggMode.PREFILL else self._d]

    def get(self, worker_id):
        return self._p if worker_id == "p1" else self._d


@pytest.mark.asyncio
async def test_mixed_request_transport_pair_is_refused():
    """A pair whose legs registered for different request transports cannot be
    dispatched: refuse before either leg is sent or accounted for, on both the
    policy-driven and the gateway-driven entry point."""
    sent = []

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={})

    class _Nats:
        async def admit(self, worker_id):
            return True

        async def stream(self, worker_id, payload):
            sent.append(worker_id)
            yield (TYPE_DONE, 200, b"")

    r = DisaggRouter(_MixedTransportPool(), _FakePolicy(), nats_client=_Nats())
    r._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    resp = await r.dispatch({"model": "m"}, stream=False)
    direct = await r.dispatch_direct(
        {"model": "m"},
        stream=False,
        path="/v1/chat/completions",
        prefill_id="p1",
        decode_id="d1",
    )

    assert resp.status_code == 503
    assert direct.status_code == 503
    assert sent == []
    assert r.policy.finished == 0
    await r.aclose()
