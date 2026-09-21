###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""PD engine abort helpers and client-disconnect abort on the stream path."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.pd_abort import (
    abort_engine_request,
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


class _FakePolicy:
    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        pass


class _FakePool:
    def list_active(self, model=None, mode=None):
        return []


def _w(wid):
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        disagg_mode=DisaggMode.PREFILL if wid.startswith("p") else DisaggMode.DECODE,
        request_transport="http",
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
    await r.aclose()
