###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Wire-level integration: save-side CopyFree (zero-copy SET) over UDS.

Boots a KvdServer with a SharedArena, connects a real KvdClient, then
exercises the lease + commit two-phase set:

    client.set_lease(size)
        -> server allocates a slot, returns (lease_token, writable mv)
    engine writes bytes directly into the shared mmap slot via mv
    client.commit_set(lease, key, length, ...)
        -> server stamps the slot header stable and links key→slot
    client.get(key) / client.get_view(key)
        -> bytes come back identical to what we wrote

Also covers SetCancel and the no-arena-fallback wire shape.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from infera.kvd.client import KvdClient
from infera.kvd.server import KvdServer
from infera.kvd.shared_arena import SharedArena
from infera.kvd.store import HostStore


@pytest.fixture
async def kvd_server_with_arena(tmp_path: Path):
    """KvdServer + SharedArena (64 KiB) on a tmp socket."""
    socket = tmp_path / f"kvd-zc-{uuid.uuid4().hex[:8]}.sock"
    arena = SharedArena(64 * 1024, pin_memory=False)
    store = HostStore(max_bytes=1 << 20, shared_arena=arena)
    server = KvdServer(socket_path=socket, max_bytes=1 << 20, store=store)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-zc-test")
    await asyncio.sleep(0)
    yield server, socket, arena
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
    arena.close()


@pytest.mark.asyncio
async def test_lease_commit_round_trip_matches_bytes(kvd_server_with_arena):
    """End-to-end: lease, write bytes into the shared mmap directly,
    commit, then read back via the regular get path — bytes match."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        assert client.shared_arena_negotiated is True

        payload = b"copy-free!" + b"\x00" * 200
        lease_result = await client.set_lease(len(payload))
        assert lease_result is not None
        lease_token, mv = lease_result
        assert lease_token > 0
        # The memoryview points into the client's local mmap and
        # must be writable.
        assert isinstance(mv, memoryview)
        assert not mv.readonly
        mv[: len(payload)] = payload

        accepted, reason = await client.commit_set(
            lease_token, b"zk1", len(payload), retention="short"
        )
        assert accepted is True
        assert reason == ""

        # Read back via the standard API — bytes are identical.
        got = await client.get(b"zk1")
        assert got == payload


@pytest.mark.asyncio
async def test_cancel_set_releases_slot(kvd_server_with_arena):
    """SetCancel drops the lease — a subsequent SetReserve picks up
    the slot. The server doesn't index any key for the cancelled
    lease."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        r1 = await client.set_lease(128)
        assert r1 is not None
        lease1, _ = r1

        await client.cancel_set(lease1)

        # No entries got published.
        from infera.kvd.wire import StatsResponse  # noqa: F401

        stats = await client.stats()
        assert stats.entries == 0

        # Another lease succeeds.
        r2 = await client.set_lease(128)
        assert r2 is not None
        lease2, _ = r2
        # Distinct tokens — leases are never reused.
        assert lease2 != lease1


@pytest.mark.asyncio
async def test_no_arena_server_rejects_lease(tmp_path):
    """A server without a shared arena (or a client that opted out)
    must refuse lease requests cleanly — client.set_lease returns
    None and the engine falls back to the legacy Set path."""
    socket = tmp_path / "kvd-zc-no-arena.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0)
    try:
        async with KvdClient(socket, prefer_shared_arena=True) as client:
            assert client.shared_arena_negotiated is False
            result = await client.set_lease(128)
            assert result is None
    finally:
        server.shutdown()
        try:
            await asyncio.wait_for(serve_task, timeout=2.0)
        except asyncio.TimeoutError:
            serve_task.cancel()


@pytest.mark.asyncio
async def test_lease_oversize_rejected(kvd_server_with_arena):
    """Once the arena's slot grid is locked at the first reserve, a
    bigger lease size is refused (client.set_lease returns None)."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        r1 = await client.set_lease(100)
        assert r1 is not None
        lease, mv = r1
        mv[:8] = b"firstput"
        await client.commit_set(lease, b"first", 8)

        # Slot grid is now locked at ceil(100+16, 64) = 128.
        # 200 bytes (216 inc header) won't fit.
        oversize = await client.set_lease(200)
        assert oversize is None


@pytest.mark.asyncio
async def test_cancel_on_disconnect_releases_leases(tmp_path):
    """The server's connection-close cleanup must cancel every
    outstanding lease the dropped connection held — otherwise a
    worker crash leaks arena slots forever."""
    socket = tmp_path / f"kvd-zc-dc-{uuid.uuid4().hex[:8]}.sock"
    arena = SharedArena(64 * 1024, pin_memory=False)
    store = HostStore(max_bytes=1 << 20, shared_arena=arena)
    server = KvdServer(socket_path=socket, max_bytes=1 << 20, store=store)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0)
    try:
        # Connect, hold three leases, then close abruptly.
        client = KvdClient(socket, prefer_shared_arena=True)
        await client.connect()
        leases = []
        for _ in range(3):
            r = await client.set_lease(64)
            assert r is not None
            leases.append(r[0])

        assert arena.stats().reservations_active == 3

        # Hard close — the server's read loop sees IncompleteReadError
        # / ConnectionResetError and runs cleanup.
        await client.close()

        # Give the server a moment to run finally-block cleanup.
        for _ in range(50):
            if arena.stats().reservations_active == 0:
                break
            await asyncio.sleep(0.01)

        stats = arena.stats()
        assert stats.reservations_active == 0
        assert stats.reservations_cancelled >= 3
    finally:
        server.shutdown()
        try:
            await asyncio.wait_for(serve_task, timeout=2.0)
        except asyncio.TimeoutError:
            serve_task.cancel()
        arena.close()
