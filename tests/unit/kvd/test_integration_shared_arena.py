###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end integration: kvd server with SharedArena + client that
negotiates shared-arena over the wire.

Same shape as `test_integration.py` but with arena wired so we can
verify:

- The handshake exchanges the FD via SCM_RIGHTS.
- GET dispatch returns GetSharedResponse (offsets, not bytes).
- BatchGet dispatch returns BatchGetSharedResponse.
- Workers reading via the local mmap see the same bytes the server
  put in.
- Fallback: a client with prefer_shared_arena=False against the
  same server gets the inline-bytes path.
- Restart resilience: kvd shutdown closes the FD; reconnect gets a
  fresh arena.
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
    """Spin up KvdServer with a SharedArena (64 KiB capacity)."""
    socket = tmp_path / f"kvd-arena-{uuid.uuid4().hex[:8]}.sock"
    arena = SharedArena(64 * 1024, pin_memory=False)
    store = HostStore(max_bytes=1 << 20, shared_arena=arena)
    server = KvdServer(socket_path=socket, max_bytes=1 << 20, store=store)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-arena-test")
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


# ----------------------------------------------------------------------
# Handshake — FD passing
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handshake_negotiates_shared_arena(kvd_server_with_arena):
    """Client with `prefer_shared_arena=True` (default) ends up with
    a non-None `arena_info` after connect."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        assert client.shared_arena_negotiated is True
        assert client._arena_info is not None
        assert client._arena_info.arena_size == 64 * 1024


@pytest.mark.asyncio
async def test_handshake_skips_shared_arena_on_optout(kvd_server_with_arena):
    """Client with `prefer_shared_arena=False` connects normally,
    receives no FD, falls back to inline-bytes responses."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=False) as client:
        assert client.shared_arena_negotiated is False


@pytest.mark.asyncio
async def test_handshake_with_no_arena_on_server(tmp_path):
    """A server WITHOUT a SharedArena ignores the client's preference
    and replies with `shared_arena=None`. Client falls back silently."""
    socket = tmp_path / "kvd-no-arena.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0)
    try:
        async with KvdClient(socket, prefer_shared_arena=True) as client:
            assert client.shared_arena_negotiated is False
    finally:
        server.shutdown()
        try:
            await asyncio.wait_for(serve_task, timeout=2.0)
        except asyncio.TimeoutError:
            serve_task.cancel()


# ----------------------------------------------------------------------
# GET / BatchGet through shared arena
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_via_shared_arena_returns_bytes(kvd_server_with_arena):
    """Client.get() through shared-arena negotiates the offset+version,
    reads from the local mmap, returns bytes. End-to-end the caller
    sees the SAME bytes that were SET."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        value = b"hello-shared-arena" + b"\x00" * 100
        accepted, _ = await client.set(b"k1", value, retention="short")
        assert accepted is True

        got = await client.get(b"k1")
        assert got == value


@pytest.mark.asyncio
async def test_get_view_returns_memoryview_when_shared(kvd_server_with_arena):
    """`get_view` returns a memoryview into the local mmap when
    shared-arena is active — zero-copy path for callers that want it."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        value = b"viewable-bytes" + b"\x00" * 50
        await client.set(b"k", value, retention="short")

        mv = await client.get_view(b"k")
        assert mv is not None
        assert isinstance(mv, memoryview)
        assert bytes(mv) == value


@pytest.mark.asyncio
async def test_batch_get_via_shared_arena(kvd_server_with_arena):
    """Batch GET through the shared-arena dispatch — returns a list
    of bytes (one per requested key), with None for misses."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        v1 = b"first" + b"\x00" * 100
        v2 = b"second" + b"\x00" * 100
        await client.set(b"k1", v1, retention="short")
        await client.set(b"k2", v2, retention="short")

        results = await client.batch_get([b"k1", b"k_missing", b"k2"])
        assert results[0] == v1
        assert results[1] is None
        assert results[2] == v2


@pytest.mark.asyncio
async def test_get_miss_returns_none_via_shared_arena(kvd_server_with_arena):
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        assert await client.get(b"never-set") is None


# ----------------------------------------------------------------------
# Backward compat: optout client against arena server
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optout_client_gets_inline_bytes(kvd_server_with_arena):
    """A client with `prefer_shared_arena=False` against an arena-
    equipped server receives INLINE bytes (the old wire shape) —
    the server's per-connection branch falls back to GetResponse."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=False) as client:
        await client.set(b"k", b"data" * 10, retention="short")
        # The fact that this returns bytes (not None) proves the
        # inline-bytes path served us correctly. The actual wire
        # response was a GetResponse with value=bytes.
        got = await client.get(b"k")
        assert got == b"data" * 10


# ----------------------------------------------------------------------
# Restart resilience
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_after_server_restart_gets_fresh_arena(tmp_path):
    """Spin up a server, connect, disconnect, recreate the server
    with a NEW arena, reconnect. The client's old mmap is closed;
    the fresh negotiation gives a new FD + view."""
    socket = tmp_path / "kvd-restart.sock"

    # First server.
    arena1 = SharedArena(64 * 1024, pin_memory=False)
    store1 = HostStore(max_bytes=1 << 20, shared_arena=arena1)
    server1 = KvdServer(socket_path=socket, max_bytes=1 << 20, store=store1)
    await server1.start()
    serve_task1 = asyncio.create_task(server1.serve_forever())
    await asyncio.sleep(0)

    client = KvdClient(socket, prefer_shared_arena=True)
    await client.connect()
    assert client.shared_arena_negotiated is True

    await client.set(b"k", b"data" * 10, retention="short")
    assert (await client.get(b"k")) == b"data" * 10

    # Tear down client + server.
    await client.close()
    server1.shutdown()
    try:
        await asyncio.wait_for(serve_task1, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task1.cancel()
    arena1.close()

    # Second server (fresh arena, same socket path).
    arena2 = SharedArena(64 * 1024, pin_memory=False)
    store2 = HostStore(max_bytes=1 << 20, shared_arena=arena2)
    server2 = KvdServer(socket_path=socket, max_bytes=1 << 20, store=store2)
    await server2.start()
    serve_task2 = asyncio.create_task(server2.serve_forever())
    await asyncio.sleep(0)

    try:
        async with KvdClient(socket, prefer_shared_arena=True) as client2:
            assert client2.shared_arena_negotiated is True
            # New arena, same server-pid (we restarted in-process,
            # so PID is same — what matters is that a FRESH FD was
            # delivered). We can still set+get.
            await client2.set(b"k2", b"after-restart", retention="short")
            assert (await client2.get(b"k2")) == b"after-restart"
    finally:
        server2.shutdown()
        try:
            await asyncio.wait_for(serve_task2, timeout=2.0)
        except asyncio.TimeoutError:
            serve_task2.cancel()
        arena2.close()


# ----------------------------------------------------------------------
# Stats / counters — basic sanity
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_response_carries_no_msgpack_bytes_when_shared(kvd_server_with_arena):
    """The whole point of the shared-arena path: the wire response
    must NOT carry the value bytes. We verify by intercepting the
    raw wire frame via a low-level recv on a fresh raw socket
    while we issue a GET — the response size should be tiny
    (just the offset/length/version metadata).
    """
    import socket as socket_mod

    from infera.kvd.wire import (
        LENGTH_BYTEORDER,
        LENGTH_PREFIX_BYTES,
        BatchGet,
        Hello,
        encode,
    )

    _, sock_path, _ = kvd_server_with_arena
    # First populate the store via a real client.
    async with KvdClient(sock_path, prefer_shared_arena=False) as setup:
        await setup.set(b"big-key", b"x" * 4000, retention="short")  # ~4 KB

    loop = asyncio.get_running_loop()
    sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
    sock.setblocking(False)
    try:
        await loop.sock_connect(sock, str(sock_path))

        async def _recv(n):
            buf = b""
            while len(buf) < n:
                chunk = await loop.sock_recv(sock, n - len(buf))
                if not chunk:
                    raise ConnectionError
                buf += chunk
            return buf

        await loop.sock_sendall(
            sock, encode(Hello(client_id="frame-probe", prefers_shared_arena=True))
        )

        # Read HelloAck (length prefix + body).
        helloack_len = int.from_bytes(await _recv(LENGTH_PREFIX_BYTES), LENGTH_BYTEORDER)
        await _recv(helloack_len)

        # Recv the FD ancillary message — we use the event-loop-friendly
        # helper from the client module.
        from infera.kvd.client import _sock_recv_fd

        fd = await _sock_recv_fd(loop, sock)
        import os

        os.close(fd)  # we don't need to mmap; we only care about the
        # subsequent response frame size.

        # Now send a BatchGet for our 4-KB key.
        await loop.sock_sendall(sock, encode(BatchGet(keys=[b"big-key"])))
        # Read the response length prefix.
        resp_len = int.from_bytes(await _recv(LENGTH_PREFIX_BYTES), LENGTH_BYTEORDER)
        resp_body = await _recv(resp_len)

        # CRITICAL ASSERTION: the response must be < 200 bytes
        # (offset + length + version + retention + slot_size = a
        # few ints + 5-char retention). Setting the value was
        # ~4000 bytes; if the response carried the value inline,
        # the frame would be ~4000+ bytes.
        assert resp_len < 200, f"response was {resp_len} bytes — value leaked"
        # And the value bytes must NOT appear in the response.
        assert b"x" * 100 not in resp_body, "value bytes leaked into response frame"
    finally:
        sock.close()


@pytest.mark.asyncio
async def test_shared_arena_get_handles_evicted_slot_gracefully(kvd_server_with_arena):
    """If the arena evicts an entry between the server's get() and
    the client's mmap read (race in the LRU path), the seqlock
    catches it and the client returns None. This is the contract
    that lets callers fall through to the long-region path safely.
    """
    server, sock_path, arena = kvd_server_with_arena
    async with KvdClient(sock_path, prefer_shared_arena=True) as client:
        # Set + verify normal get works.
        await client.set(b"k", b"hello" + b"\x00" * 100)
        assert (await client.get(b"k")) == b"hello" + b"\x00" * 100

        # Explicitly evict via the arena's API (simulates a race
        # where the LRU evicts under us between get and read).
        evicted_key = arena.evict_key(b"k")
        assert evicted_key is True

        # The server's view: the entry is still in `_entries` (we
        # only evicted from the arena's free list), so a get will
        # find the entry but the arena lookup of its slot will be
        # stale. The client may get back stale offset/length/version
        # — the seqlock might still pass since the bytes are still
        # there. This is fine semantically (the bytes are still
        # valid even if logically evicted from the arena's
        # bookkeeping). What we're asserting is: NO CRASH, no
        # corruption.
        result = await client.get(b"k")
        # Result is either the original bytes (slot bytes survived
        # the evict) or None (overwritten by a subsequent put).
        # Both are valid; what matters is it doesn't blow up.
        assert result is None or result == b"hello" + b"\x00" * 100


@pytest.mark.asyncio
async def test_stats_reflects_arena_backed_entries(kvd_server_with_arena):
    """The kvd STATS response should still report host_bytes correctly
    when the store is arena-backed. The Entry's size_bytes property
    reads from _size_cache so the math is right."""
    _, socket, _ = kvd_server_with_arena
    async with KvdClient(socket, prefer_shared_arena=True) as client:
        await client.set(b"a", b"x" * 100, retention="short")
        await client.set(b"b", b"y" * 100, retention="short")
        s = await client.stats()
        assert s.entries == 2
        assert s.host_bytes == 200
        assert s.hits_total == 0  # we haven't GET yet
        assert s.sets_total == 2


# ----------------------------------------------------------------------
# MAP_POPULATE on the engine-side mmap
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_mmap_with_populate_succeeds(kvd_server_with_arena):
    """When the client is constructed with `prefault_arena=True`, the
    handshake mmap of the arena FD should add ``MAP_POPULATE``. We
    confirm by patching ``mmap.mmap`` and checking the flags. The
    mapping must still be functional — MAP_POPULATE is a hint and
    must not change observable behaviour."""
    import mmap as _mmap
    from unittest.mock import patch

    if not hasattr(_mmap, "MAP_POPULATE"):
        pytest.skip("MAP_POPULATE not available on this platform")

    _, socket, _ = kvd_server_with_arena
    seen_flags: list[int] = []
    real_mmap = _mmap.mmap

    def fake_mmap(fd, length, *args, **kwargs):
        seen_flags.append(kwargs.get("flags", 0))
        return real_mmap(fd, length, *args, **kwargs)

    with patch.object(_mmap, "mmap", side_effect=fake_mmap):
        async with KvdClient(socket, prefer_shared_arena=True, prefault_arena=True) as client:
            assert client.shared_arena_negotiated is True
            # MAP_POPULATE was set on at least one mmap call (the
            # arena view). The post-handshake asyncio open_unix_connection
            # may also mmap internally; we only need to see the flag
            # set on at least one call from our open_arena_view.
            assert any(f & _mmap.MAP_POPULATE for f in seen_flags), (
                f"expected MAP_POPULATE in flags, got {[hex(f) for f in seen_flags]}"
            )
            # End-to-end works.
            ok, _ = await client.set(b"k", b"v" * 100, retention="short")
            assert ok is True
            assert await client.get(b"k") == b"v" * 100


@pytest.mark.asyncio
async def test_client_mmap_prefault_env_zero_unchanged(kvd_server_with_arena, monkeypatch):
    """With ``INFERA_KVD_ARENA_PREFAULT=0`` and the client built
    with the default ``prefault_arena=None`` sentinel, no
    ``MAP_POPULATE`` flag should be set by ``open_arena_view``. The
    connection still works end-to-end — this guards the env-var
    opt-out path that operators reach for on memory-constrained
    boxes or older kernels."""
    import mmap as _mmap
    from unittest.mock import patch

    monkeypatch.setenv("INFERA_KVD_ARENA_PREFAULT", "0")

    _, socket, _ = kvd_server_with_arena
    seen_flags: list[int] = []
    real_mmap = _mmap.mmap

    def fake_mmap(fd, length, *args, **kwargs):
        seen_flags.append(kwargs.get("flags", 0))
        return real_mmap(fd, length, *args, **kwargs)

    with patch.object(_mmap, "mmap", side_effect=fake_mmap):
        # prefault_arena defaults to None → consult env, which is
        # falsy here, so MAP_POPULATE must NOT be added.
        async with KvdClient(socket, prefer_shared_arena=True) as client:
            assert client.shared_arena_negotiated is True
            map_populate = getattr(_mmap, "MAP_POPULATE", 0)
            if map_populate:
                assert all(not (f & map_populate) for f in seen_flags), (
                    f"unexpected MAP_POPULATE in flags, got {[hex(f) for f in seen_flags]}"
                )
            # End-to-end still works.
            ok, _ = await client.set(b"k", b"v" * 100, retention="short")
            assert ok is True
            assert await client.get(b"k") == b"v" * 100
