###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Migration as the client experiences it.

`test_migration.py` covers the accounting; this drives the dispatch path, where
the property that matters is what reaches the client. A migration the client can
detect -- a gap, a repetition, an error frame -- has failed at its only job,
however correct the bookkeeping was.
"""

from __future__ import annotations

import json

import pytest

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.mixed import MixedRouter
from infera.router.policy.target import RouteTarget


def _w(wid: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport="nats",
        disagg_mode=DisaggMode.MIXED,
    )


class _Pool:
    def __init__(self, workers):
        self._workers = workers

    def list_active(self, model=None, mode=None):
        return list(self._workers)


class _Policy:
    """Round-robins, so a migration lands somewhere other than the failure."""

    def __init__(self):
        self.picks: list[str] = []

    def pick(self, candidates, body, role_hint=None):
        target = RouteTarget(candidates[0])
        self.picks.append(target.worker.worker_id)
        return target, []

    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        pass


def chunk(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()


class _ScriptedNats:
    """Replays a scripted reply per worker, recording the bodies it was sent."""

    def __init__(self, scripts: dict[str, list[tuple]]):
        self.scripts = scripts
        self.bodies: dict[str, dict] = {}
        self.sent: dict[str, dict] = {}

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        self.sent[worker_id] = payload
        self.bodies[worker_id] = payload["body"]
        assert "path" in payload
        for item in self.scripts.get(worker_id, []):
            yield item


def _router(nats, workers, *, limit: int) -> MixedRouter:
    return MixedRouter(_Pool(workers), _Policy(), nats_client=nats, migration_limit=limit)


async def _collect(response) -> bytes:
    out = b""
    async for piece in response.body_iterator:
        out += piece if isinstance(piece, bytes) else piece.encode()
    return out


@pytest.mark.asyncio
async def test_a_broken_stream_continues_on_another_worker():
    """The client reads one uninterrupted answer across a worker failure."""
    nats = _ScriptedNats(
        {
            # w1 produces two tokens, then its stream breaks.
            "w1": [
                (TYPE_DATA, None, chunk("Hello")),
                (TYPE_DATA, None, chunk(", ")),
                (TYPE_ERROR, None, b"worker vanished"),
            ],
            # w2 continues, and finishes.
            "w2": [
                (TYPE_DATA, None, chunk("world")),
                (TYPE_DONE, 200, b""),
            ],
        }
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        stream=True,
    )
    body = await _collect(resp)

    assert b"Hello" in body and b", " in body and b"world" in body
    assert b"error" not in body, "the client must not see the failure"
    # Nothing is repeated: the second worker continued rather than restarted.
    assert body.count(b"Hello") == 1
    await r.aclose()


@pytest.mark.asyncio
async def test_the_second_worker_is_asked_to_continue_not_to_restart():
    """What the next worker receives is the original request plus what the
    client already read, with the budget reduced by what it cost."""
    nats = _ScriptedNats(
        {
            "w1": [
                (TYPE_DATA, None, chunk("one")),
                (TYPE_DATA, None, chunk(" two")),
                (TYPE_ERROR, None, b"gone"),
            ],
            "w2": [(TYPE_DATA, None, chunk(" three")), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "count"}], "max_tokens": 9},
        stream=True,
    )
    await _collect(resp)

    sent = nats.bodies["w2"]
    assert sent["messages"][-1] == {"role": "assistant", "content": "one two"}
    assert sent["max_tokens"] == 7, "two tokens were already spent"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_failure_the_client_never_sees_is_still_visible_when_unmigratable():
    """With nowhere to go the stream ends with an error rather than silently
    stopping -- a truncated answer with no explanation is worse."""
    nats = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("partial")), (TYPE_ERROR, None, b"gone")]})
    r = _router(nats, [_w("w1")], limit=1)  # no second worker
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    body = await _collect(resp)

    assert b"partial" in body
    assert b"error" in body
    await r.aclose()


@pytest.mark.asyncio
async def test_migration_is_off_by_default():
    """It changes what a worker is asked to produce, so it is opt-in."""
    nats = _ScriptedNats(
        {"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_ERROR, None, b"gone")], "w2": []}
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=0)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    body = await _collect(resp)

    assert b"error" in body
    assert "w2" not in nats.bodies, "nothing may be dispatched when migration is off"
    await r.aclose()


def exact_chunk(content: str, ids: list[int], prompt_ids: list[int] | None = None) -> bytes:
    obj = {"choices": [{"delta": {"content": content}, "token_ids": ids}]}
    if prompt_ids is not None:
        obj["prompt_token_ids"] = prompt_ids
    return f"data: {json.dumps(obj)}\n\n".encode()


def completion_chunk(text: str, ids: list[int], prompt_ids: list[int] | None = None) -> bytes:
    obj = {"choices": [{"text": text, "token_ids": ids}], "object": "text_completion"}
    if prompt_ids is not None:
        obj["prompt_token_ids"] = prompt_ids
    return f"data: {json.dumps(obj)}\n\n".encode()


def _vllm(wid: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.VLLM,
        request_transport="nats",
        disagg_mode=DisaggMode.MIXED,
    )


@pytest.mark.asyncio
async def test_an_exact_continuation_resumes_from_the_sampled_ids():
    """The second worker is handed the ids the model actually produced, not a
    re-encoding of the words, so no token boundary can shift."""
    nats = _ScriptedNats(
        {
            "w1": [
                (TYPE_DATA, None, exact_chunk("Hel", [100], prompt_ids=[1, 2])),
                (TYPE_DATA, None, exact_chunk("lo", [200])),
                (TYPE_ERROR, None, b"gone"),
            ],
            "w2": [
                (TYPE_DATA, None, completion_chunk(" there", [300])),
                (TYPE_DONE, 200, b""),
            ],
        }
    )
    r = _router(nats, [_vllm("w1"), _vllm("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        stream=True,
    )
    body = await _collect(resp)

    sent = nats.sent["w2"]
    assert sent["body"]["prompt"] == [1, 2, 100, 200], "prompt ids + everything sampled"
    assert sent["path"] == "/v1/completions", "the only endpoint taking token ids"
    assert sent["body"]["max_tokens"] == 8, "two real tokens spent"
    assert b"Hel" in body and b"lo" in body and b" there" in body
    assert b"error" not in body
    await r.aclose()


@pytest.mark.asyncio
async def test_the_client_keeps_reading_chat_across_an_exact_migration():
    """The continuation is fetched from the completions endpoint, which answers
    in a different shape. The client asked for chat and must keep getting it."""
    nats = _ScriptedNats(
        {
            "w1": [
                (TYPE_DATA, None, exact_chunk("start", [1], prompt_ids=[9])),
                (TYPE_ERROR, None, b"gone"),
            ],
            "w2": [(TYPE_DATA, None, completion_chunk(" end", [2])), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router(nats, [_vllm("w1"), _vllm("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    body = await _collect(resp)

    frames = [json.loads(ln[6:]) for ln in body.split(b"\n") if ln.startswith(b"data: ")]
    assert all("delta" in f["choices"][0] for f in frames), "every frame is chat-shaped"
    assert all("text" not in f["choices"][0] for f in frames)
    assert [f["choices"][0]["delta"]["content"] for f in frames] == ["start", " end"]
    await r.aclose()


@pytest.mark.asyncio
async def test_the_token_ids_never_reach_the_client():
    """The router asks for them for its own use. Leaving them in would make the
    response depend on an operator's migration setting."""
    nats = _ScriptedNats(
        {"w1": [(TYPE_DATA, None, exact_chunk("hi", [5], prompt_ids=[1])), (TYPE_DONE, 200, b"")]}
    )
    r = _router(nats, [_vllm("w1")], limit=1)
    body = await _collect(
        await r.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )
    assert b"token_ids" not in body
    assert b"hi" in body
    await r.aclose()


@pytest.mark.asyncio
async def test_ids_are_requested_only_when_a_migration_could_use_them():
    nats = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_DONE, 200, b"")]})
    r = _router(nats, [_vllm("w1")], limit=1)
    await _collect(
        await r.dispatch({"model": "m", "messages": [{"role": "u", "content": "h"}]}, stream=True)
    )
    assert nats.sent["w1"]["body"]["return_token_ids"] is True
    await r.aclose()

    off = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_DONE, 200, b"")]})
    r_off = _router(off, [_vllm("w1")], limit=0)
    await _collect(
        await r_off.dispatch(
            {"model": "m", "messages": [{"role": "u", "content": "h"}]}, stream=True
        )
    )
    assert "return_token_ids" not in off.sent["w1"]["body"], (
        "an engine must not be asked for work no migration will use"
    )
    await r_off.aclose()


@pytest.mark.asyncio
async def test_sglang_chat_streams_are_never_asked_for_ids():
    """SGLang rejects the request outright, so asking would turn every
    migratable chat request into a 400."""
    nats = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_DONE, 200, b"")]})
    r = _router(nats, [_w("w1")], limit=1)  # _w builds an SGLANG worker
    await _collect(
        await r.dispatch({"model": "m", "messages": [{"role": "u", "content": "h"}]}, stream=True)
    )
    assert "return_token_ids" not in nats.sent["w1"]["body"]
    await r.aclose()


@pytest.mark.asyncio
async def test_an_engine_that_reports_no_ids_still_migrates_on_text():
    """The exact path is an optimisation. Losing it must cost precision, not
    the migration itself."""
    nats = _ScriptedNats(
        {
            "w1": [(TYPE_DATA, None, chunk("half")), (TYPE_ERROR, None, b"gone")],
            "w2": [(TYPE_DATA, None, chunk(" done")), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=1)
    body = await _collect(
        await r.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )
    sent = nats.sent["w2"]
    assert sent["path"] == "/v1/chat/completions", "no ids means no endpoint change"
    assert sent["body"]["messages"][-1] == {"role": "assistant", "content": "half"}
    assert b"half" in body and b" done" in body and b"error" not in body
    await r.aclose()


@pytest.mark.asyncio
async def test_a_draining_worker_hands_its_stream_over():
    """A worker leaving on purpose looks like a broken one to the client -- the
    stream continues either way -- but not to an operator, so the two are
    counted separately."""
    from infera.common.nats_request import DRAINING_NOTICE
    from infera.server import metrics

    before = metrics.migrations_total.labels(reason="worker_draining")._value.get()
    nats = _ScriptedNats(
        {
            "w1": [(TYPE_DATA, None, chunk("half")), (TYPE_ERROR, None, DRAINING_NOTICE)],
            "w2": [(TYPE_DATA, None, chunk(" done")), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    body = await _collect(resp)

    assert b"half" in body and b" done" in body
    assert b"error" not in body
    after = metrics.migrations_total.labels(reason="worker_draining")._value.get()
    assert after == before + 1, "a planned handover is not counted as a fault"
    await r.aclose()


@pytest.mark.asyncio
async def test_the_worker_is_told_whether_the_stream_can_be_resumed():
    """The worker cannot know on its own: handing a stream back early is only
    safe because the router promised to continue it."""
    nats = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_DONE, 200, b"")]})
    r = _router(nats, [_w("w1")], limit=1)
    await _collect(
        await r.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )
    assert nats.sent["w1"]["migratable"] is True
    await r.aclose()

    nats_off = _ScriptedNats({"w1": [(TYPE_DATA, None, chunk("x")), (TYPE_DONE, 200, b"")]})
    r_off = _router(nats_off, [_w("w1")], limit=0)
    await _collect(
        await r_off.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )
    assert nats_off.sent["w1"]["migratable"] is False, (
        "a worker must not shed streams this router cannot resume"
    )
    await r_off.aclose()


@pytest.mark.asyncio
async def test_a_tool_call_ends_the_stream_rather_than_replaying_it():
    """The client holds half a tool call. Resuming would send it a whole one
    after that, so the stream ends visibly instead."""
    from infera.server import metrics

    before = metrics.migrations_failed_total.labels(reason="poisoned")._value.get()
    tool = json.dumps(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]}
    )
    nats = _ScriptedNats(
        {
            "w1": [
                (TYPE_DATA, None, chunk("checking")),
                (TYPE_DATA, None, f"data: {tool}\n\n".encode()),
                (TYPE_ERROR, None, b"gone"),
            ],
            "w2": [(TYPE_DATA, None, chunk("whatever")), (TYPE_DONE, 200, b"")],
        }
    )
    r = _router(nats, [_vllm("w1"), _vllm("w2")], limit=1)
    body = await _collect(
        await r.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )

    assert b"error" in body
    assert "w2" not in nats.sent, "a request that cannot be rebuilt must not be resumed"
    after = metrics.migrations_failed_total.labels(reason="poisoned")._value.get()
    assert after == before + 1, "the lost capability has to be visible to an operator"
    await r.aclose()


@pytest.mark.asyncio
async def test_a_draining_worker_that_cannot_be_replaced_says_so():
    """Ending with 'failed' would report a planned shutdown as a fault."""
    from infera.common.nats_request import DRAINING_NOTICE

    nats = _ScriptedNats(
        {"w1": [(TYPE_DATA, None, chunk("half")), (TYPE_ERROR, None, DRAINING_NOTICE)]}
    )
    r = _router(nats, [_vllm("w1")], limit=1)  # nowhere else to go
    body = await _collect(
        await r.dispatch(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
        )
    )
    assert b"shutting down" in body
    assert b"failed" not in body
    await r.aclose()


@pytest.mark.asyncio
async def test_the_limit_bounds_how_often_one_generation_moves():
    """Two workers that both break, with a budget of one move: the second
    failure ends the stream instead of starting a third attempt."""
    nats = _ScriptedNats(
        {
            "w1": [(TYPE_DATA, None, chunk("a")), (TYPE_ERROR, None, b"gone")],
            "w2": [(TYPE_DATA, None, chunk("b")), (TYPE_ERROR, None, b"gone too")],
        }
    )
    r = _router(nats, [_w("w1"), _w("w2")], limit=1)
    resp = await r.dispatch(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, stream=True
    )
    body = await _collect(resp)

    assert b"a" in body and b"b" in body
    assert b"error" in body, "the budget is spent; the failure now reaches the client"
    await r.aclose()
