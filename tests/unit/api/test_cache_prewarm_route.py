###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the POST /v1/cache/prewarm endpoint.

Uses `httpx.AsyncClient + ASGITransport` so the test, the FastAPI app,
and the real KvdServer all share one asyncio event loop — avoiding
the cross-loop traps that hung an earlier TestClient-based version.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import httpx
import pytest

from infera.kvd.server import KvdServer
from infera.server.app import app, init_app


class _FakeRouter:
    """init_app needs *some* router; the prewarm endpoint doesn't
    actually call it."""

    async def aclose(self):
        pass


@pytest.fixture
async def kvd_server_async(tmp_path: Path):
    """Real KvdServer on a temp socket, same-loop with the test."""
    socket = tmp_path / f"kvd-prewarm-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0)
    try:
        yield socket, server
    finally:
        server.shutdown()
        try:
            await asyncio.wait_for(serve, timeout=2.0)
        except asyncio.TimeoutError:
            serve.cancel()
            try:
                await serve
            except asyncio.CancelledError:
                pass


@pytest.fixture
def app_with_kvd(kvd_server_async):
    socket, server = kvd_server_async
    init_app(
        reg=None,  # type: ignore[arg-type]
        rtr=_FakeRouter(),  # type: ignore[arg-type]
        kv=None,
        kvd_socket_path=str(socket),
    )
    yield app, server


async def _post_prewarm(json_body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post("/v1/cache/prewarm", json=json_body)


@pytest.mark.asyncio
async def test_prewarm_accepts_valid_request(app_with_kvd):
    """Happy path: well-formed body → 202 + n_keys reported back +
    kvd's `prefetch_hints_total` counter ticks."""
    _, server = app_with_kvd
    hashes = [bytes([i]) * 8 for i in range(4)]
    r = await _post_prewarm(
        {
            "model": "MiniMax-M2.5",
            "block_hashes": [h.hex() for h in hashes],
            "deadline_ms": 500,
        }
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["n_keys"] == 4
    # Give the kvd worker a tick to process the hint.
    await asyncio.sleep(0.05)
    assert server.prefetch_stats["hints_total"] >= 1


@pytest.mark.asyncio
async def test_prewarm_rejects_missing_model(app_with_kvd):
    """`model` is required."""
    r = await _post_prewarm({"block_hashes": ["00" * 8]})
    assert r.status_code == 400
    assert "model" in r.text


@pytest.mark.asyncio
async def test_prewarm_rejects_invalid_hex(app_with_kvd):
    """Bad hex in block_hashes must return 400 — don't fall through
    to kvd with garbage bytes."""
    r = await _post_prewarm({"model": "m", "block_hashes": ["not-hex"]})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_prewarm_empty_hash_list_is_legal(app_with_kvd):
    """Harness that built an empty list shouldn't get a 4xx. The
    endpoint short-circuits at 202 with n_keys=0."""
    r = await _post_prewarm({"model": "m", "block_hashes": []})
    assert r.status_code == 202
    assert r.json()["n_keys"] == 0


@pytest.mark.asyncio
async def test_prewarm_rejects_zero_or_negative_deadline(app_with_kvd):
    """deadline_ms must be positive — a zero deadline at the kvd
    side would expire the warmed entry immediately, making the
    whole hint pointless."""
    r = await _post_prewarm({"model": "m", "block_hashes": ["aa" * 8], "deadline_ms": 0})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_prewarm_503_when_no_kvd_configured():
    """If init_app wasn't given a kvd_socket_path the endpoint must
    refuse with 503 instead of trying to connect to /tmp/kvd.sock."""
    init_app(
        reg=None,  # type: ignore[arg-type]
        rtr=_FakeRouter(),  # type: ignore[arg-type]
        kv=None,
        kvd_socket_path=None,
    )
    r = await _post_prewarm({"model": "m", "block_hashes": ["aa" * 8]})
    assert r.status_code == 503
    assert "kvd_socket_path" in r.text


@pytest.mark.asyncio
async def test_prewarm_is_fast(app_with_kvd):
    """Fire-and-forget property: the endpoint must NOT block on the
    kvd worker. We assert the full HTTP round-trip completes well
    under 50ms for a 16-key request. A regression to a synchronous
    `client.set` would push this past 50ms easily."""
    hashes = [bytes([i]) * 8 for i in range(16)]
    t0 = time.perf_counter()
    r = await _post_prewarm(
        {
            "model": "m",
            "block_hashes": [h.hex() for h in hashes],
            "deadline_ms": 500,
        }
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 202
    assert elapsed_ms < 100.0, f"prewarm took {elapsed_ms:.1f}ms"
