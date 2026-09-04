###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The startup check that kv-aware routing is actually going to work.

Every test here is really the same assertion: a router whose render disagrees
with the engine's must SAY SO, because nothing else will. That failure produces
no error, no dropped request and no unhealthy worker — only a hit rate of zero,
which looks exactly like a cold cache.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.render_probe import (
    PROBE_BODIES,
    ProbeResult,
    engine_render_variant,
    probe_worker,
    spawn_probe,
)
from infera.router.kv_event.render_variant import RenderVariant, VariantRegistry

_BODIES = {"plain": {"messages": [{"role": "user", "content": "hi"}]}}


class _Hasher:
    """Stands in for BlockHasher; `token_ids_for` is the whole contract."""

    def __init__(self, ids: list[int] | None = None) -> None:
        self._ids = ids if ids is not None else [1, 2, 3]
        self.seen: list[dict] = []

    def token_ids_for(self, body, *, engine=None):
        self.seen.append(body)
        return self._ids


def _worker(**kw) -> WorkerInfo:
    return WorkerInfo(
        worker_id=kw.pop("worker_id", "w1"),
        url=kw.pop("url", "http://w1:30000"),
        model_name=kw.pop("model_name", "glm53"),
        engine=kw.pop("engine", EngineType.SGLANG),
        **kw,
    )


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_agreement_is_confirmed():
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1, 2, 3], "count": 3}))
    got = await probe_worker(_Hasher([1, 2, 3]), _worker(), bodies=_BODIES, client=client)
    assert got.ok is True
    await client.aclose()


@pytest.mark.asyncio
async def test_divergence_is_reported_with_the_offending_position():
    """The message has to name the token index, not just "mismatch". The first
    thing an operator needs to know is whether we diverge in the preamble (a
    template/kwargs problem, every prompt affected) or deep in the
    conversation (a message-shape problem)."""
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1, 9, 3]}))
    got = await probe_worker(_Hasher([1, 2, 3]), _worker(), bodies=_BODIES, client=client)
    assert got.ok is False
    assert "diverges at token 1" in got.detail
    await client.aclose()


@pytest.mark.asyncio
async def test_a_truncated_render_is_a_divergence():
    """Equal prefixes but different lengths still means every block hash from
    the end onward is wrong — and `zip` alone would call this a match."""
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1, 2, 3, 4]}))
    got = await probe_worker(_Hasher([1, 2, 3]), _worker(), bodies=_BODIES, client=client)
    assert got.ok is False
    assert "diverges at token 3" in got.detail
    await client.aclose()


@pytest.mark.asyncio
async def test_no_tokenize_endpoint_is_unknown_not_failure():
    """An engine that doesn't serve the endpoint tells us nothing. Calling that
    a divergence would page on every such fleet and train people to ignore the
    alert that matters."""
    client = _client(lambda r: httpx.Response(404))
    got = await probe_worker(_Hasher(), _worker(), bodies=_BODIES, client=client)
    assert got.ok is None
    await client.aclose()


@pytest.mark.asyncio
async def test_falls_back_to_the_unprefixed_alias():
    paths: list[str] = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/tokenize":
            return httpx.Response(404)
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    client = _client(handler)
    got = await probe_worker(_Hasher([1, 2, 3]), _worker(), bodies=_BODIES, client=client)
    assert got.ok is True
    assert paths == ["/v1/tokenize", "/tokenize"]
    await client.aclose()


@pytest.mark.asyncio
async def test_an_unreachable_worker_never_raises():
    """This runs from a registration hook. A probe that throws would turn a
    slow worker into a broken router."""

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    got = await probe_worker(_Hasher(), _worker(), bodies=_BODIES, client=client)
    assert got.ok is None
    await client.aclose()


@pytest.mark.asyncio
async def test_a_router_that_can_render_nothing_is_unknown_not_diverged():
    """Declining EVERY body says nothing about this worker -- it is a router
    with no usable tokenizer, a deployment choice rather than a fault. Only a
    PARTIAL decline is a statement about the worker (see the next test)."""

    class _Declining(_Hasher):
        def token_ids_for(self, body, *, engine=None):
            return None

    calls: list[str] = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"tokens": [1]})

    client = _client(handler)
    got = await probe_worker(_Declining(), _worker(), bodies=_BODIES, client=client)
    assert got.ok is None
    assert "declined" in got.detail
    assert calls == []  # and it didn't bother the engine to find out
    await client.aclose()


@pytest.mark.asyncio
async def test_one_declined_body_among_matching_ones_still_fails():
    """The partial case is the dangerous one: `plain` matches, `tools` does not
    render, and the worker used to come back Confirmed."""
    declined = {"tools"}

    class _PartlyDeclining(_Hasher):
        def token_ids_for(self, body, *, engine=None):
            return None if body.get("tools") else [1, 2, 3]

    client = _client(lambda r: httpx.Response(200, json={"tokens": [1, 2, 3]}))
    bodies = {"plain": {"messages": []}, "tools": {"messages": [], "tools": [{"x": 1}]}}
    got = await probe_worker(_PartlyDeclining(), _worker(), bodies=bodies, client=client)
    assert got.ok is False, "a matching body must not outvote an unhashable one"
    assert "tools" in got.detail
    await client.aclose()
    assert declined  # keeps the intent legible


@pytest.mark.asyncio
async def test_probe_bodies_carry_the_workers_model():
    """The hasher keys its tokenizer off `model`; a probe without one would
    check nothing at all."""
    hasher = _Hasher([1])
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1]}))
    await probe_worker(hasher, _worker(model_name="glm53"), bodies=_BODIES, client=client)
    assert [b["model"] for b in hasher.seen] == ["glm53"]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_responses_probe_body_is_hashed_as_chat():
    """`/v1/tokenize` runs `_process_messages`. Sending `input` would 400 or
    compare a different path than the one the engine hashes after `_make_request`."""
    pytest.importorskip("sglang.srt.entrypoints.openai.serving_responses")
    hasher = _Hasher([1])
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1]}))
    got = await probe_worker(
        hasher,
        _worker(),
        bodies={"responses": {"input": "What is 2+2?"}},
        client=client,
    )
    assert got.ok is True
    assert "messages" in hasher.seen[0]
    assert "input" not in hasher.seen[0]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_missing_responses_converter_does_not_diverge_when_chat_matches(
    monkeypatch,
):
    """Router hosts without sglang still hash chat. `to_chat_body` then returns
    None for the Responses probe body; treating that as a decline used to mark
    the worker Diverged while chat kv-aware was fine."""
    monkeypatch.setattr(
        "infera.router.kv_event.responses_input.to_chat_body",
        lambda body: None,
    )
    hasher = _Hasher([1])
    client = _client(lambda r: httpx.Response(200, json={"tokens": [1]}))
    got = await probe_worker(
        hasher,
        _worker(),
        bodies={
            "plain": {"messages": [{"role": "user", "content": "hi"}]},
            "responses": {"input": "What is 2+2?"},
        },
        client=client,
    )
    assert got.ok is True
    assert [b.get("input") for b in hasher.seen] == [None]
    await client.aclose()


@pytest.mark.asyncio
async def test_one_bad_body_among_good_ones_still_fails():
    """Divergence is usually conditional — tools render fine, a tool-call turn
    doesn't. Passing on the majority would miss exactly the agentic traffic
    that kv-aware is bought for."""

    class _PerBody(_Hasher):
        def token_ids_for(self, body, *, engine=None):
            return [7] if body.get("tools") else [1]

    def handler(request):
        return httpx.Response(200, json={"tokens": [1]})

    client = _client(handler)
    got = await probe_worker(_PerBody(), _worker(), bodies=PROBE_BODIES, client=client)
    assert got.ok is False
    assert "tools" in got.detail and "tool_call" in got.detail
    await client.aclose()


# ----------------------------------------------------------------------
# The server-side template defaults
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_variant_is_applied_to_what_we_render_but_not_to_what_we_ask():
    """The engine merges its defaults itself; sending them back would hide a
    divergence by making the request agree with the router's guess."""
    hasher = _Hasher([1])
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"tokens": [1]})

    client = _client(handler)
    variant = RenderVariant.from_default_chat_template_kwargs({"reasoning_effort": "high"})
    got = await probe_worker(hasher, _worker(), bodies=_BODIES, client=client, variant=variant)
    assert got.ok is True
    assert hasher.seen[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in sent[0]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"default_chat_template_kwargs": {"reasoning_effort": "high"}},
        {"server_args": {"default_chat_template_kwargs": {"reasoning_effort": "high"}}},
    ],
    ids=["flat", "nested"],
)
async def test_server_info_is_read_flat_or_under_server_args(payload):
    # sglang has served this both ways across the versions we run.
    client = _client(lambda r: httpx.Response(200, json=payload))
    got = await engine_render_variant(client, _worker())
    assert got is not None and got.label() == 'reasoning_effort="high"'
    await client.aclose()


@pytest.mark.asyncio
async def test_a_worker_that_cannot_be_asked_is_none_not_empty():
    """`None` keeps the router's existing assumption; an empty variant would
    silently overwrite a correct --kv-default-chat-template-kwargs with
    "this worker has none"."""
    client = _client(lambda r: httpx.Response(404))
    assert await engine_render_variant(client, _worker()) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_a_worker_that_reports_no_defaults_gets_the_empty_variant():
    """Explicitly `null`/`{}` means the engine HAS none -- record that, so it
    wins over a fleet flag that does not apply to this worker."""
    for payload in ({"default_chat_template_kwargs": None}, {"default_chat_template_kwargs": {}}):
        client = _client(lambda r, p=payload: httpx.Response(200, json=p))
        got = await engine_render_variant(client, _worker())
        assert got is not None and got.is_empty()
        await client.aclose()


@pytest.mark.asyncio
async def test_an_engine_that_does_not_report_the_field_is_not_an_empty_variant():
    """The distinction the caller acts on. An engine older than the flag, a
    renamed key, or a proxy that trims the payload answers 200 without the
    field -- that is "could not ask", and returning the empty variant would
    have it RECORDED, where `for_worker` prefers it over the fleet default and
    silently discards the operator's --kv-default-chat-template-kwargs for
    every worker and every request."""
    client = _client(lambda r: httpx.Response(200, json={"server_args": {"tp_size": 8}}))
    got = await engine_render_variant(client, _worker())
    assert got is None
    await client.aclose()


async def test_a_model_already_ruled_out_is_not_probed(monkeypatch):
    """The Python mirror of the Rust probe's `is_enabled` gate.

    Every body would come back "router declined to render" and the verdict
    would be Unknown before the engine was asked anything, so the only thing
    left of the probe is the /get_server_info round trip.

    Async on purpose. `spawn_probe` also returns early when there is no running
    loop, so a synchronous version of this test passes with the gate deleted --
    it never reaches the code it claims to be about.
    """
    calls = []
    monkeypatch.setattr(
        "infera.router.kv_event.render_probe.probe_worker",
        lambda *a, **k: calls.append(a),
    )

    fresh = BlockHasher()
    spawn_probe(fresh, _worker(model_name="m"), report=lambda *a: None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls, "a model nothing has ruled out must still be probed"

    calls.clear()
    ruled_out = BlockHasher()
    # What `_get_tokenizer` writes when a load fails.
    ruled_out._tokenizers[(EngineType.SGLANG, "m")] = None
    spawn_probe(ruled_out, _worker(model_name="m"), report=lambda *a: None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == []


def test_the_gate_answers_not_known_to_be_broken(monkeypatch):
    """It must never trigger a load: it runs on the discovery hook, where a
    blocking `from_pretrained` would stall registration for every worker."""
    hasher = BlockHasher()
    monkeypatch.setattr(
        BlockHasher, "_load", lambda *a, **k: pytest.fail("can_render must not load")
    )
    assert hasher.can_render("never-seen", EngineType.SGLANG)


# ---- probe reservations ---------------------------------------------------


class _Policy:
    """The claim/record bookkeeping, isolated from the rest of the policy."""

    def __init__(self):
        from infera.router.policy.kv_event_aware import KvEventAwarePolicy

        self.p = KvEventAwarePolicy.__new__(KvEventAwarePolicy)
        self.p._parity_pending = {}
        self.p._parity_labels = {}
        self.p._parity_verdict = {}
        self.p._parity_epoch = 0
        self.p._variants = VariantRegistry()

    def claim(self) -> int | None:
        from infera.router.policy.kv_event_aware import _UNPROBED

        if "w1" in self.p._parity_pending:
            return None
        settled = self.p._parity_verdict.get("w1", _UNPROBED)
        if settled is not _UNPROBED and settled is not None:
            return None
        self.p._parity_epoch += 1
        self.p._parity_pending["w1"] = self.p._parity_epoch
        return self.p._parity_epoch


def test_a_stale_probe_cannot_consume_a_replacement_probes_reservation():
    """Probe #1 in flight -> worker leaves -> rejoins -> probe #2 claims.

    Keyed by worker id alone, probe #1's report consumed probe #2's
    reservation: the dead instance's verdict was written, the fresh one was
    dropped as "left the fleet", and every later claim was refused.
    """
    pol = _Policy()
    stale = pol.claim()
    pol.p._parity_pending.clear()  # the worker departed
    fresh = pol.claim()
    assert stale != fresh

    pol.p._report_render_parity(_worker(), ProbeResult(False, "x"), stale)
    assert pol.p._parity_pending.get("w1") == fresh, "probe #2 must keep its reservation"
    assert "w1" not in pol.p._parity_verdict, "the dead instance's verdict must not land"


def test_an_unknown_verdict_is_retried_but_a_settled_one_is_not():
    """`None` means the probe never reached the engine -- a worker still
    starting, a transient 503. Treating it as final pins -1 for the life of the
    router on a worker that is now perfectly answerable."""
    pol = _Policy()
    e = pol.claim()
    pol.p._report_render_parity(_worker(), ProbeResult(None, "engine did not answer"), e)
    again = pol.claim()
    assert again is not None, "an unreachable engine must be asked again"

    pol.p._report_render_parity(_worker(), ProbeResult(True, "ok"), again)
    assert pol.claim() is None, "a settled verdict stays settled"


@pytest.mark.asyncio
async def test_an_abandoned_probe_gives_its_reservation_back():
    """`spawn_probe` returns early for a non-sglang engine. The caller has
    already claimed, so without `release` that worker is in-flight forever."""
    released: list[tuple[str, int]] = []
    w = _worker()
    w.engine = EngineType.VLLM
    spawn_probe(
        _Hasher(),
        w,
        report=lambda *a: None,
        epoch=7,
        release=lambda wid, e: released.append((wid, e)),
    )
    assert released == [(w.worker_id, 7)]
