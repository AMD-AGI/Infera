###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Graceful drain on the HTTP transport.

On NATS infera owns the request path and knows what is in flight. On HTTP the
router talks straight to the engine, so infera has to ask the engine — which
means every failure mode is a *measurement* failure, and the interesting cases
are all about what happens when the number cannot be trusted.

The rule these tests pin down: never treat "unknown" as "idle". An unreadable
metric that defaults to zero would make the drain pass instantly and cut live
generations, and it would do so silently.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from infera.common.engine_metrics import inflight_from_metrics, metric_name, parse_metric
from infera.common.worker_pool import EngineType
from infera.engine import drain as drain_mod
from infera.engine.drain import drain_engine_inflight

# --- reading the engine's numbers ---------------------------------------------


def test_parses_plain_and_labelled_gauges():
    assert parse_metric("vllm:num_requests_running 3.0\n", "vllm:num_requests_running") == 3.0
    assert parse_metric('x:g{a="b"} 7\n', "x:g") == 7.0


def test_missing_series_is_none_not_zero():
    """The distinction the whole drain rests on."""
    assert parse_metric("something_else 1\n", "vllm:num_requests_running") is None


def test_inflight_counts_running_plus_waiting():
    """A queued request is work a client is waiting on; killing the process
    loses it just as surely as one mid-generation."""
    text = "vllm:num_requests_running 2\nvllm:num_requests_waiting 5\n"
    assert inflight_from_metrics(text, EngineType.VLLM) == 7.0


def test_unknown_engine_yields_none():
    """ATOM has no mapping. Guessing one would report an idle engine, and an
    idle engine is exactly the answer that makes a drain cut live requests."""
    assert metric_name("requests_running", EngineType.ATOM) is None
    assert inflight_from_metrics("vllm:num_requests_running 4\n", EngineType.ATOM) is None


def test_partial_metrics_still_count():
    text = "sglang:num_running_reqs 1\n"
    assert inflight_from_metrics(text, EngineType.SGLANG) == 1.0


# --- the drain loop -----------------------------------------------------------


def _patch_client(monkeypatch, handler):
    """Route drain's httpx client at a mock transport."""
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), timeout=5.0)

    monkeypatch.setattr(drain_mod.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_returns_when_engine_goes_idle(monkeypatch):
    counts = iter([3, 2, 0])

    def handler(request):
        return httpx.Response(200, text=f"vllm:num_requests_running {next(counts)}\n")

    _patch_client(monkeypatch, handler)
    drained = await drain_engine_inflight(
        host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=5, poll_interval=0.01
    )
    assert drained is True


@pytest.mark.asyncio
async def test_times_out_while_still_busy(monkeypatch):
    def handler(request):
        return httpx.Response(200, text="vllm:num_requests_running 4\n")

    _patch_client(monkeypatch, handler)
    drained = await drain_engine_inflight(
        host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=0.2, poll_interval=0.01
    )
    assert drained is False, "a busy engine must not report a clean drain"


@pytest.mark.asyncio
async def test_unreadable_metric_does_not_hang(monkeypatch, caplog):
    """A rolling update that stalls on a parse failure is worse than one that
    cuts a request -- and a silent full-timeout wait is indistinguishable from
    a genuinely busy worker."""

    def handler(request):
        return httpx.Response(200, text="totally_different_metric 1\n")

    _patch_client(monkeypatch, handler)
    started = asyncio.get_running_loop().time()
    drained = await drain_engine_inflight(
        host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=30, poll_interval=0.01
    )
    elapsed = asyncio.get_running_loop().time() - started
    assert drained is False
    assert elapsed < 1.0, f"returned after {elapsed:.1f}s; must not wait out the timeout"
    assert "WITHOUT draining" in caplog.text


@pytest.mark.asyncio
async def test_unreachable_engine_does_not_hang(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_client(monkeypatch, handler)
    drained = await drain_engine_inflight(
        host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=30, poll_interval=0.01
    )
    assert drained is False


@pytest.mark.asyncio
async def test_zero_timeout_is_a_no_op(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text="vllm:num_requests_running 0\n")

    _patch_client(monkeypatch, handler)
    assert (
        await drain_engine_inflight(host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=0)
        is False
    )
    assert calls == [], "--drain-timeout 0 must not even probe"


@pytest.mark.asyncio
async def test_never_raises_on_the_shutdown_path(monkeypatch):
    """This runs immediately before engine.stop(); an exception here would skip
    the teardown that follows."""

    def factory(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(drain_mod.httpx, "AsyncClient", factory)
    assert (
        await drain_engine_inflight(host="1.2.3.4", port=8000, engine=EngineType.VLLM, timeout=1)
        is False
    )
