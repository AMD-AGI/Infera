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

import httpx
import pytest

from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.kv_event.render_probe import PROBE_BODIES, probe_worker

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
async def test_a_body_the_router_declines_is_not_a_divergence():
    """`token_ids_for` returning None means the router already knows it can't
    reproduce this body and will route it on load — a known gap, honestly
    handled. The probe exists to find the cases where we are confidently wrong."""

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
    assert calls == []  # and it didn't bother the engine to find out
    await client.aclose()


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
