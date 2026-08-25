###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Re-anchoring the KV-event chain on startup.

The router's cache index can only be built forward from the single event whose
``parent_block_hash`` is ``None``, and the engine publishes that one during
warmup — before anything is subscribed. Whoever tails the stream therefore joins
to a chain whose anchor is already gone, and every event after it names a parent
that was never seen and is dropped. Nothing errors; kv-aware just degrades to
routing on load while ``/health`` stays green.

Flushing the cache is what repairs it, because ``AllBlocksCleared`` is the only
event that rebuilds an anchor from nothing. The subtlety these tests pin down is
that issuing the flush is not the same as repairing anything: ZMQ ``connect()``
is asynchronous, so a flush sent before the subscription attached is lost the
same way the original anchor was — and leaves a worker that looks repaired and
is not. So the loop waits for the *observed* clear, not for its own POST.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from infera.common.worker_pool import EngineType
from infera.engine import flush as flush_mod
from infera.engine.flush import anchor_kv_chain, flush_engine_prefix_cache


def _patch_client(monkeypatch, handler):
    """Route flush's httpx client at a mock transport."""
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), timeout=5.0)

    monkeypatch.setattr(flush_mod.httpx, "AsyncClient", factory)


# --- the POST itself ----------------------------------------------------------


@pytest.mark.asyncio
async def test_posts_the_engines_own_flush_endpoint(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        return httpx.Response(200, text="Cache flushed.")

    _patch_client(monkeypatch, handler)
    assert await flush_engine_prefix_cache(host="0.0.0.0", port=30000, engine=EngineType.SGLANG)
    # 0.0.0.0 is what the engine binds, but it is not a destination: the worker
    # has to reach its own engine over loopback.
    assert seen == [("POST", "http://127.0.0.1:30000/flush_cache")]


@pytest.mark.asyncio
async def test_vllm_uses_its_own_endpoint_name(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    _patch_client(monkeypatch, handler)
    assert await flush_engine_prefix_cache(host="10.0.0.1", port=8000, engine=EngineType.VLLM)
    assert seen == ["http://10.0.0.1:8000/reset_prefix_cache"]


@pytest.mark.asyncio
async def test_a_busy_engine_refuses_and_says_so(monkeypatch, caplog):
    """SGLang gates the flush on an idle scheduler and answers 400 otherwise.

    Reporting that as failure is the point: a silent 200-regardless would make a
    refused flush indistinguishable from a successful one, and the caller would
    register a worker whose chain is still dead.
    """
    _patch_client(monkeypatch, lambda r: httpx.Response(400, text="not idle"))
    with caplog.at_level(logging.INFO, logger="infera.engine.flush"):
        assert not await flush_engine_prefix_cache(
            host="127.0.0.1", port=30000, engine=EngineType.SGLANG
        )
    assert "not idle" in caplog.text


@pytest.mark.asyncio
async def test_an_unreachable_engine_never_raises(monkeypatch):
    """This runs next to registration on the startup path. An exception here
    would cost the worker its registration to fix a routing optimisation."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _patch_client(monkeypatch, handler)
    assert not await flush_engine_prefix_cache(
        host="127.0.0.1", port=30000, engine=EngineType.SGLANG
    )


@pytest.mark.asyncio
async def test_atom_is_not_flushed(monkeypatch):
    """ATOM has no anchor to lose. Its hook reports ``parent_block_hash=None``
    for the first block of every sequence, so the chain re-roots on the next
    cold prefix, and it emits no clear event a flush could produce anyway. The
    guessed ``/flush_cache`` only ever bought a 404 -- which the router's
    self-heal reads as a busy worker and backs off from."""
    called = []
    _patch_client(monkeypatch, lambda r: called.append(1) or httpx.Response(200))
    assert not await flush_engine_prefix_cache(host="127.0.0.1", port=1, engine=EngineType.ATOM)
    assert not called


@pytest.mark.asyncio
async def test_an_engine_with_no_known_endpoint_is_skipped(monkeypatch):
    called = []
    _patch_client(monkeypatch, lambda r: called.append(1) or httpx.Response(200))
    assert not await flush_engine_prefix_cache(host="127.0.0.1", port=1, engine="mystery-engine")
    assert not called, "guessing an endpoint would POST at an unknown handler"


# --- the closed loop ----------------------------------------------------------


@pytest.mark.asyncio
async def test_stops_as_soon_as_the_clear_is_observed(monkeypatch):
    posts = []
    observed = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        # A live subscription sees the resulting event.
        observed.set()
        return httpx.Response(200)

    _patch_client(monkeypatch, handler)
    assert await anchor_kv_chain(
        host="127.0.0.1", port=30000, engine=EngineType.SGLANG, observed=observed
    )
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_retries_while_the_flush_lands_on_nobody(monkeypatch):
    """The failure the retry exists for.

    The relay's ``start()`` has returned but its SUB has not attached yet, so
    the engine flushes, publishes the clear, and nobody is there — exactly the
    race that lost the anchor in the first place. Waiting on the observation
    rather than on a fixed sleep is what tells the two apart.
    """
    posts = []
    observed = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        if len(posts) >= 3:  # the subscription is finally live
            observed.set()
        return httpx.Response(200)

    _patch_client(monkeypatch, handler)
    assert await anchor_kv_chain(
        host="127.0.0.1",
        port=30000,
        engine=EngineType.SGLANG,
        observed=observed,
        settle=0.01,
    )
    assert len(posts) == 3


@pytest.mark.asyncio
async def test_gives_up_loudly_rather_than_blocking_registration(monkeypatch, caplog):
    """A worker whose chain cannot be anchored still has to serve traffic."""
    _patch_client(monkeypatch, lambda r: httpx.Response(200))
    with caplog.at_level(logging.WARNING, logger="infera.engine.flush"):
        assert not await anchor_kv_chain(
            host="127.0.0.1",
            port=30000,
            engine=EngineType.SGLANG,
            observed=asyncio.Event(),
            attempts=2,
            settle=0.01,
        )
    assert "load alone" in caplog.text


@pytest.mark.asyncio
async def test_the_whole_loop_is_bounded_not_just_each_attempt(monkeypatch):
    """This is awaited immediately before ``register()``.

    Every second spent here is a second the worker exists and cannot be routed
    to, so an engine that answers slowly must not be able to turn a routing
    optimisation into a startup stall of attempts x (timeout + settle).
    """
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return httpx.Response(200)

    _patch_client(monkeypatch, handler)
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert not await anchor_kv_chain(
        host="127.0.0.1",
        port=30000,
        engine=EngineType.SGLANG,
        observed=asyncio.Event(),
        attempts=50,
        settle=0.05,
        deadline=0.3,
    )
    assert loop.time() - started < 2.0, "the deadline must cut the loop short"
    assert len(posts) < 50, "and it must stop retrying, not merely stop waiting"


@pytest.mark.asyncio
async def test_an_already_anchored_chain_is_left_alone(monkeypatch):
    """Nothing is broken, so nothing is flushed -- the cache is real GPU memory
    and discarding it to fix a problem the worker does not have is pure loss."""
    called = []
    _patch_client(monkeypatch, lambda r: called.append(1) or httpx.Response(200))
    observed = asyncio.Event()
    observed.set()
    assert await anchor_kv_chain(
        host="127.0.0.1", port=30000, engine=EngineType.SGLANG, observed=observed
    )
    assert not called


@pytest.mark.asyncio
async def test_the_deadline_covers_the_posts_not_only_the_waiting(monkeypatch):
    """A wedged engine is exactly what the deadline is for, and it is the POST
    that hangs, not the wait. Checking the budget only afterwards let the last
    attempt overrun by a whole ``post_timeout`` -- with the defaults, ~18s
    against a 15s bound."""
    timeouts = []

    async def never_answers(*, host, port, engine, timeout):
        timeouts.append(timeout)
        await asyncio.sleep(timeout)
        return False

    monkeypatch.setattr(flush_mod, "flush_engine_prefix_cache", never_answers)
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert not await anchor_kv_chain(
        host="127.0.0.1",
        port=30000,
        engine=EngineType.SGLANG,
        observed=asyncio.Event(),
        attempts=10,
        settle=0.05,
        post_timeout=0.2,
        deadline=0.3,
    )
    elapsed = loop.time() - started
    assert elapsed < 0.55, f"overran the deadline: {elapsed:.2f}s"
    # No POST is handed budget the loop does not have left.
    assert sum(timeouts) <= 0.3 + 1e-6, timeouts
