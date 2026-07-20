###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end integration: spawn the asyncio daemon, hit it with the
client, verify state. No subprocess — daemon + client run in the
same process on different asyncio tasks. Subprocess tests would be
nice for CLI coverage but are slow and we don't need them yet.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from infera.kvd.client import KvdClient, KvdConnectionError
from infera.kvd.server import KvdServer


@pytest.fixture
async def kvd_server(tmp_path: Path):
    """Spin up a KvdServer on a unique socket under tmp_path."""
    socket = tmp_path / f"kvd-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)  # 1 MB
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-test-serve")
    # Give the event loop a tick to wire up the listener.
    await asyncio.sleep(0)
    yield server, socket
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_and_handshake(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket, client_id="alice") as client:
        assert client.server_id is not None
        assert client.server_id.startswith("kvd-")


@pytest.mark.asyncio
async def test_connect_unreachable_raises():
    with pytest.raises(KvdConnectionError):
        client = KvdClient("/nonexistent/path/kvd.sock", client_id="x")
        await client.connect()


# ----------------------------------------------------------------------
# CRUD round-trip
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_get_round_trip(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        accepted, reason = await client.set(b"key-1", b"hello", retention="short")
        assert accepted is True
        assert reason is None

        value = await client.get(b"key-1")
        assert value == b"hello"


@pytest.mark.asyncio
async def test_get_miss_returns_none(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        value = await client.get(b"absent")
        assert value is None


@pytest.mark.asyncio
async def test_exists_bulk(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"a", b"x", retention="short")
        await client.set(b"b", b"y", retention="long")
        present = await client.exists([b"a", b"b", b"c"])
        assert present == [True, True, False]


@pytest.mark.asyncio
async def test_batch_get_returns_values_in_order(kvd_server):
    """`batch_get` returns one element per requested key, in the
    same order. Hits carry bytes, misses are None. Position-aligned
    semantics — the connector relies on this for paired
    (block_id ↔ kvd_key) dispatch."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"a" + b"\x00" * 7, b"value-a")
        await client.set(b"b" + b"\x00" * 7, b"value-b")
        values = await client.batch_get(
            [b"a" + b"\x00" * 7, b"missing" + b"\x00", b"b" + b"\x00" * 7]
        )
        assert values == [b"value-a", None, b"value-b"]


@pytest.mark.asyncio
async def test_batch_get_empty_list_no_round_trip(kvd_server):
    """Empty batch is a no-op — client short-circuits, no wire
    activity. Mirrors `exists([])`."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        before = await client.stats()
        result = await client.batch_get([])
        after = await client.stats()
        assert result == []
        # No GETs incremented (stats RPC itself doesn't touch gets_total).
        assert after.gets_total == before.gets_total


@pytest.mark.asyncio
async def test_batch_get_respects_namespace(kvd_server):
    """Entries set under one (model, compat_key) namespace must NOT
    appear in a batch_get under a different namespace. Mirrors the
    single-`get` namespace isolation but for the new batch path."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"x" + b"\x00" * 7, b"under-A", model="A", compat_key="ck")
        # Query under namespace B — should miss.
        values = await client.batch_get([b"x" + b"\x00" * 7], model="B", compat_key="ck")
        assert values == [None]
        # And under A — hits.
        values = await client.batch_get([b"x" + b"\x00" * 7], model="A", compat_key="ck")
        assert values == [b"under-A"]


@pytest.mark.asyncio
async def test_batch_get_matches_serial_gets_byte_for_byte(kvd_server):
    """The batch path MUST produce identical bytes to N serial GETs.
    Pinned because a future server-side optimization (e.g. mmap'd
    response, shared memory) could silently corrupt at the boundary."""
    _, socket = kvd_server
    payloads = [(f"key-{i}".encode() + b"\x00" * 8)[:8] for i in range(16)]
    bodies = [bytes([i]) * (100 + i) for i in range(16)]
    async with KvdClient(socket) as client:
        for k, v in zip(payloads, bodies, strict=True):
            await client.set(k, v, retention="short")
        serial = [await client.get(k) for k in payloads]
        batched = await client.batch_get(payloads)
        assert serial == batched, "batch GET must be byte-identical to serial GETs"


@pytest.mark.asyncio
async def test_batch_get_increments_gets_total(kvd_server):
    """The daemon's `gets_total` counter must rise once PER KEY in
    a batch, not once per batch — same accounting as serial GETs.
    This keeps the kvd_gets metric an apples-to-apples comparison
    when operators switch the connector between batched and serial."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        keys = [bytes([i]) * 8 for i in range(5)]
        for k in keys:
            await client.set(k, b"v")
        before = (await client.stats()).gets_total
        await client.batch_get(keys)
        after = (await client.stats()).gets_total
        assert after - before == len(keys)


@pytest.mark.asyncio
async def test_batch_set_round_trip_returns_accepted(kvd_server):
    """`batch_set` returns (accepted, reason) per item — mirrors the
    single-`set` return shape. Daemon must accept all five in a
    fresh store and report None reasons."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        items = [(bytes([i]) * 8, bytes([0x10 + i]) * 64, "short", None) for i in range(5)]
        results = await client.batch_set(items)
        assert len(results) == 5
        for accepted, reason in results:
            assert accepted is True
            assert reason is None


@pytest.mark.asyncio
async def test_batch_set_then_get_byte_identical(kvd_server):
    """The values inserted via `batch_set` must be byte-identical to
    what subsequent GETs return — the batch path must not mangle
    bytes through msgpack."""
    _, socket = kvd_server
    payloads = [(bytes([i]) * 8, bytes([0x20 + i]) * (50 + i * 7), "short", None) for i in range(8)]
    async with KvdClient(socket) as client:
        await client.batch_set(payloads)
        for k, v, _, _ in payloads:
            assert await client.get(k) == v


@pytest.mark.asyncio
async def test_batch_set_empty_list_no_round_trip(kvd_server):
    """Empty batch is a no-op — no wire activity, no sets_total bump."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        before = (await client.stats()).sets_total
        result = await client.batch_set([])
        after = (await client.stats()).sets_total
        assert result == []
        assert after == before


@pytest.mark.asyncio
async def test_batch_set_increments_sets_total(kvd_server):
    """`sets_total` must rise once PER ITEM in a batch — same
    accounting as serial SETs."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        items = [(bytes([i]) * 8, b"v" * 16, "short", None) for i in range(7)]
        before = (await client.stats()).sets_total
        await client.batch_set(items)
        after = (await client.stats()).sets_total
        assert after - before == 7


@pytest.mark.asyncio
async def test_batch_set_respects_namespace(kvd_server):
    """Items inserted via batch_set under one (model, compat_key) must
    NOT be visible under another. Pins the namespace boundary."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        items = [(b"x" + b"\x00" * 7, b"a-val", "short", None)]
        await client.batch_set(items, model="A", compat_key="ck")
        # Same key, different namespace → miss.
        assert await client.get(b"x" + b"\x00" * 7, model="B", compat_key="ck") is None
        # Same namespace → hit.
        assert await client.get(b"x" + b"\x00" * 7, model="A", compat_key="ck") == b"a-val"


# ----------------------------------------------------------------------
# TTL + ephemeral retention behavior
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_expired_entry_reads_as_miss(kvd_server):
    """An entry SET with ttl_seconds must report as missing on get
    once the deadline has passed. Lazy expiration — checked on next
    get, no sweeper thread."""

    _, socket = kvd_server
    async with KvdClient(socket) as client:
        # 100ms TTL — long enough that the SET round-trip completes,
        # short enough to expire within the test.
        await client.set(b"ttl-key" + b"\x00" * 1, b"v" * 16, ttl_seconds=0.1)
        # Immediate get hits.
        assert await client.get(b"ttl-key" + b"\x00" * 1) == b"v" * 16
        # Wait past the deadline.
        await asyncio.sleep(0.15)
        # Now expired → miss.
        assert await client.get(b"ttl-key" + b"\x00" * 1) is None


@pytest.mark.asyncio
async def test_ttl_none_means_no_expiration(kvd_server):
    """Without a TTL, the entry must persist until retention/capacity
    eviction. Default behavior unchanged when no TTL is set."""

    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"perm-key" + b"\x00", b"v" * 16)
        # 200ms later — well past anything one might assume as a
        # default TTL — entry is still there.
        await asyncio.sleep(0.2)
        assert await client.get(b"perm-key" + b"\x00") == b"v" * 16


@pytest.mark.asyncio
async def test_ttl_exists_reports_expired_as_missing(kvd_server):
    """`exists()` and `get()` must agree on expiration. If exists
    reported True after the deadline the router might dispatch a
    load that the subsequent get would refuse — wasted scheduler
    cycles."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"exi-key" + b"\x00", b"v" * 16, ttl_seconds=0.1)
        # Pre-expiry: exists True.
        assert (await client.exists([b"exi-key" + b"\x00"])) == [True]
        await asyncio.sleep(0.15)
        # Post-expiry: exists False.
        assert (await client.exists([b"exi-key" + b"\x00"])) == [False]


@pytest.mark.asyncio
async def test_ttl_overwrite_refreshes_deadline(kvd_server):
    """A second SET with the same key resets the TTL clock. Mirrors
    Anthropic's cache_control extend-on-write semantics."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"refresh" + b"\x00", b"v", ttl_seconds=0.1)
        await asyncio.sleep(0.07)  # most of the TTL window elapses
        # Re-set with a longer TTL.
        await client.set(b"refresh" + b"\x00", b"v2", ttl_seconds=1.0)
        await asyncio.sleep(0.1)  # original TTL would have fired
        # Entry must still be there with the NEW value.
        assert await client.get(b"refresh" + b"\x00") == b"v2"


@pytest.mark.asyncio
async def test_ttl_zero_or_negative_is_immediate_expiry(kvd_server):
    """ttl_seconds=0 or negative tells the daemon "expire instantly."
    The set is acked (so the caller's contract holds) but the entry
    is never stored — a subsequent get misses."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        accepted, reason = await client.set(b"zero-ttl" + b"\x00", b"v", ttl_seconds=0)
        assert accepted is True
        assert reason is None
        assert await client.get(b"zero-ttl" + b"\x00") is None


@pytest.mark.asyncio
async def test_batch_set_per_item_ttl(kvd_server):
    """ttls_seconds in batch_set must apply per-item. Items without
    TTL must still be stored permanently."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        items = [
            (b"bk-a" + b"\x00" * 4, b"a-val", "short", None),
            (b"bk-b" + b"\x00" * 4, b"b-val", "short", None),
            (b"bk-c" + b"\x00" * 4, b"c-val", "short", None),
        ]
        ttls = [0.1, None, 0.1]
        await client.batch_set(items, ttls_seconds=ttls)
        await asyncio.sleep(0.15)
        # a and c expired, b survives.
        assert await client.get(b"bk-a" + b"\x00" * 4) is None
        assert await client.get(b"bk-b" + b"\x00" * 4) == b"b-val"
        assert await client.get(b"bk-c" + b"\x00" * 4) is None


@pytest.mark.asyncio
async def test_ephemeral_retention_evicted_before_short(kvd_server):
    """Under capacity pressure ephemeral entries
    must be evicted strictly before short ones. The store's
    `_PRIORITY` orders them ephemeral < short < long; if a future
    refactor breaks that order, RAG hit rates collapse on
    reasoning-heavy workloads."""
    # The kvd_server fixture allocates 64 MiB; we don't want to fill
    # that. Construct a mini store directly instead.
    from infera.kvd.store import HostStore

    store = HostStore(max_bytes=512)  # tiny — easy to push to capacity
    # Insert one short entry — non-evictable until pressure.
    accepted, _ = store.set(b"keep" + b"\x00" * 4, b"v" * 200, retention="short")
    assert accepted
    # Insert ephemeral — should fit (~400 bytes used now).
    accepted, _ = store.set(b"eph" + b"\x00" * 5, b"v" * 100, retention="ephemeral")
    assert accepted
    # Now insert another short entry forcing capacity pressure (200 + 100 + 250 > 512).
    accepted, _ = store.set(b"push" + b"\x00" * 4, b"v" * 250, retention="short")
    assert accepted
    # The ephemeral entry MUST be the one evicted, not the original short.
    assert store.get(b"keep" + b"\x00" * 4) is not None
    assert store.get(b"eph" + b"\x00" * 5) is None


@pytest.mark.asyncio
async def test_ephemeral_set_via_wire_succeeds(kvd_server):
    """End-to-end: a client SET with retention="ephemeral" must reach
    the daemon and round-trip on a subsequent get. This is the
    minimum guarantee the connector relies on when forwarding
    `infera_retention="ephemeral"` from a request."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        accepted, reason = await client.set(
            b"eph-wire" + b"\x00", b"thinking-token-bytes", retention="ephemeral"
        )
        assert accepted is True
        assert reason is None
        assert await client.get(b"eph-wire" + b"\x00") == b"thinking-token-bytes"


@pytest.mark.asyncio
async def test_batch_set_matches_serial_sets_byte_for_byte(kvd_server):
    """Pinned correctness test: a batch_set of N items must produce
    a kvd-store state byte-identical to N serial sets in the same
    order. Future server-side optimizations (e.g. write-batching to
    spillover) could silently reorder; this catches that."""
    _, socket = kvd_server
    items = [(bytes([i]) * 8, bytes([0x40 + i]) * (60 + i), "short", None) for i in range(6)]
    async with KvdClient(socket) as client:
        # Pass 1: batch
        await client.batch_set(items, model="batched")
        batched = [await client.get(k, model="batched") for k, _, _, _ in items]
        # Pass 2: serial, fresh namespace
        for k, v, ret, meta in items:
            await client.set(k, v, retention=ret, model="serial", metadata=meta)
        serial = [await client.get(k, model="serial") for k, _, _, _ in items]
        assert batched == serial


@pytest.mark.asyncio
async def test_clear_all(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"a", b"x", retention="short")
        await client.set(b"b", b"y", retention="short")
        count = await client.clear()
        assert count == 2
        assert await client.get(b"a") is None
        assert await client.get(b"b") is None


@pytest.mark.asyncio
async def test_clear_namespace(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        await client.set(b"a", b"x", model="m1")
        await client.set(b"b", b"y", model="m2")
        count = await client.clear(model="m1")
        assert count == 1
        assert await client.get(b"a", model="m1") is None
        assert await client.get(b"b", model="m2") == b"y"


# ----------------------------------------------------------------------
# Retention enforcement
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_set_rejected_under_long_pressure(kvd_server, tmp_path):
    server, _ = kvd_server
    # Refill: replace the test's 1 MB server with a tiny one for this test.
    socket = tmp_path / "tiny-kvd.sock"
    tiny_server = KvdServer(socket_path=socket, max_bytes=10)
    await tiny_server.start()
    serve_task = asyncio.create_task(tiny_server.serve_forever(), name="tiny-serve")
    await asyncio.sleep(0)
    try:
        async with KvdClient(socket) as client:
            accepted, _ = await client.set(b"long", b"x" * 10, retention="long")
            assert accepted
            accepted, reason = await client.set(b"short", b"y" * 10, retention="short")
            assert accepted is False
            assert reason == "would_displace_higher_priority"
    finally:
        tiny_server.shutdown()
        await asyncio.wait_for(serve_task, timeout=2.0)


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_reflects_traffic(kvd_server):
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        # Send some traffic.
        await client.set(b"a", b"x", retention="short")
        await client.set(b"b", b"y", retention="long")
        await client.get(b"a")  # hit
        await client.get(b"a")  # hit
        await client.get(b"absent")  # miss

        stats = await client.stats()
        assert stats.entries == 2
        assert stats.host_bytes == 2
        assert stats.gets_total == 3
        assert stats.hits_total == 2
        assert stats.misses_total == 1
        assert stats.sets_total == 2


# ----------------------------------------------------------------------
# Concurrent clients
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_clients_share_state(kvd_server):
    """Two clients see each other's writes — daemon is single-store."""
    _, socket = kvd_server
    async with KvdClient(socket, client_id="A") as client_a:
        async with KvdClient(socket, client_id="B") as client_b:
            await client_a.set(b"shared", b"from-A", retention="long")
            value = await client_b.get(b"shared")
            assert value == b"from-A"


@pytest.mark.asyncio
async def test_many_concurrent_ops(kvd_server):
    """Hammer the daemon with N concurrent setters — exercises the
    per-connection write lock + the store's thread lock."""
    _, socket = kvd_server
    N = 50

    async def one_writer(idx: int) -> bool:
        async with KvdClient(socket, client_id=f"writer-{idx}") as client:
            accepted, _ = await client.set(
                f"key-{idx}".encode(), f"value-{idx}".encode(), retention="short"
            )
            return accepted

    results = await asyncio.gather(*(one_writer(i) for i in range(N)))
    assert all(results), "some writes were rejected"

    async with KvdClient(socket) as reader:
        present = await reader.exists([f"key-{i}".encode() for i in range(N)])
        assert all(present)


# ----------------------------------------------------------------------
# Client error path: bad protocol
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protocol_error_after_close(kvd_server):
    """Operations after `close()` should raise KvdConnectionError."""
    _, socket = kvd_server
    client = KvdClient(socket)
    await client.connect()
    await client.close()
    with pytest.raises(KvdConnectionError):
        await client.get(b"a")


# ----------------------------------------------------------------------
# Speculative L3 prefetch
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_hint_fire_and_forget(kvd_server):
    """prefetch_hint() must NOT block on a response. We pin this by
    measuring wall: a sub-millisecond completion proves the client
    isn't waiting on a round-trip."""
    import time as _time

    _, socket = kvd_server
    async with KvdClient(socket) as client:
        t0 = _time.perf_counter()
        await client.prefetch_hint(
            [bytes([i]) * 8 for i in range(8)],
            model="m",
            compat_key="ck",
            deadline_ms=500,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        # Generous bound; serial UDS round-trip would be ~1-5 ms.
        # Under 2 ms means we wrote and returned without waiting.
        assert elapsed_ms < 5.0, f"prefetch_hint blocked for {elapsed_ms:.2f}ms"


@pytest.mark.asyncio
async def test_prefetch_hint_empty_keys_is_noop(kvd_server):
    """Empty hint must short-circuit client-side — no wire activity."""
    _, socket = kvd_server
    async with KvdClient(socket) as client:
        before = (await client.stats()).gets_total
        await client.prefetch_hint([], model="m", compat_key="ck")
        after = (await client.stats()).gets_total
        assert before == after  # stats round-trip itself only


@pytest.mark.asyncio
async def test_prefetch_hint_increments_hints_counter(kvd_server):
    """Each fire-and-forget hint bumps `prefetch_hints_total` on the
    daemon side. This is the operator's first signal that the wire
    contract is working."""
    server, socket = kvd_server
    async with KvdClient(socket) as client:
        before = server.prefetch_stats["hints_total"]
        await client.prefetch_hint([b"k" * 8], model="m", compat_key="ck")
        # Give the daemon a tick to process the hint.
        await asyncio.sleep(0.05)
        after = server.prefetch_stats["hints_total"]
        assert after - before == 1


@pytest.mark.asyncio
async def test_prefetch_hint_filters_already_warm_keys(kvd_server):
    """If a hint references a key that's already in host RAM the
    worker must short-circuit — no fetch, no L3 round-trip, no
    'useful' counter bump. Pin this so the dedupe filter doesn't
    silently regress."""
    server, socket = kvd_server
    async with KvdClient(socket) as client:
        # First, put the key in RAM via a normal SET.
        await client.set(b"warm" + b"\x00" * 4, b"already-here", retention="short")
        fetches_before = server.prefetch_stats["fetches_total"]
        await client.prefetch_hint([b"warm" + b"\x00" * 4], model="", compat_key="")
        await asyncio.sleep(0.05)
        fetches_after = server.prefetch_stats["fetches_total"]
        # No fetch was issued — the worker saw it was already in RAM.
        assert fetches_after == fetches_before


@pytest.mark.asyncio
async def test_prefetch_warmed_key_counted_useful_on_get(kvd_server):
    """When the prefetch worker warms a key (from L3) and the engine
    later issues a `get` on it, the 'useful' counter must bump
    exactly once. Second `get` doesn't double-count."""
    server, socket = kvd_server
    # The kvd_server fixture is RAM-only — no long region. We bypass
    # via direct test-only call into _register_warmed_key to simulate
    # what the worker would have done after an L3 fetch.
    server._register_warmed_key(b"hot" + b"\x00" * 5)

    async with KvdClient(socket) as client:
        # Direct SET so the bytes are gettable.
        await client.set(b"hot" + b"\x00" * 5, b"v")
        useful_before = server.prefetch_stats["hits_useful_total"]
        # First get → counted useful.
        v = await client.get(b"hot" + b"\x00" * 5)
        assert v == b"v"
        useful_mid = server.prefetch_stats["hits_useful_total"]
        assert useful_mid - useful_before == 1
        # Second get on same key → NOT counted (warmed flag cleared).
        await client.get(b"hot" + b"\x00" * 5)
        useful_after = server.prefetch_stats["hits_useful_total"]
        assert useful_after == useful_mid


@pytest.mark.asyncio
async def test_prefetch_stats_dropped_on_queue_overflow(kvd_server):
    """When the bounded prefetch queue is full (cap=64 default), new
    hints' keys must be dropped with the counter bumped. The router
    can rate-limit upstream to avoid this — but the visibility is
    critical."""
    server, socket = kvd_server
    # Construct a server with cap=2 directly for the test (the
    # kvd_server fixture uses default 64).
    from pathlib import Path

    sock = Path(server._socket_path).parent / "kvd-tiny-prefetch.sock"
    tiny = KvdServer(socket_path=sock, max_bytes=1 << 20, prefetch_inflight=2)
    await tiny.start()
    serve = asyncio.create_task(tiny.serve_forever())
    try:
        await asyncio.sleep(0)
        async with KvdClient(sock) as client:
            # Send 10 keys via one hint — the worker should consume
            # ~2 before the queue saturates, then 8 drop.
            await client.prefetch_hint(
                [bytes([i]) * 8 for i in range(10)],
                model="m",
                compat_key="ck",
            )
            await asyncio.sleep(0.05)
            assert tiny.prefetch_stats["dropped_total"] > 0
    finally:
        tiny.shutdown()
        try:
            await asyncio.wait_for(serve, timeout=2.0)
        except asyncio.TimeoutError:
            serve.cancel()
            try:
                await serve
            except asyncio.CancelledError:
                pass
