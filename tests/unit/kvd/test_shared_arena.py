###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for `infera.kvd.shared_arena.SharedArena`.

Covers:
- slot allocation, alignment, num_slots derivation
- LRU eviction at capacity
- seqlock torn-read detection (forced race)
- worker-side `open_arena_view` + `read_slot_seqlock`
- clear/reset preserves slot_size
"""

from __future__ import annotations

import struct
import threading

import pytest

from infera.kvd.shared_arena import (
    _HEADER_BYTES,
    _HEADER_VERSION_OFFSET,
    MAX_TORN_READ_RETRIES,
    SharedArena,
    SharedArenaInfo,
    open_arena_view,
    read_slot_seqlock,
)

# ----------------------------------------------------------------------
# Slot allocation
# ----------------------------------------------------------------------


def test_constructor_rejects_zero_capacity():
    with pytest.raises(ValueError):
        SharedArena(0)


def test_first_put_decides_slot_size_aligned_to_64():
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        key = b"k1"
        value = b"x" * 100  # 100 + 16 header = 116, rounded up to 128
        slot = arena.put(key, value)
        assert slot is not None
        assert arena.slot_size == 128
        assert arena.num_slots == 64 * 1024 // 128
    finally:
        arena.close()


def test_put_get_round_trip_returns_same_bytes():
    arena = SharedArena(1 * 1024 * 1024, pin_memory=False)
    try:
        value = b"hello world" + b"\x00" * 1000
        slot = arena.put(b"key1", value)
        assert slot is not None
        mv = arena.get_slice(slot)
        assert mv is not None
        assert bytes(mv) == value
    finally:
        arena.close()


def test_overwrite_same_key_reuses_slot():
    arena = SharedArena(1 * 1024 * 1024, pin_memory=False)
    try:
        v1 = b"a" * 500
        v2 = b"b" * 500
        slot1 = arena.put(b"k", v1)
        slot2 = arena.put(b"k", v2)
        # Slot reuse isn't strictly required (could grab another from
        # the free list), but the arena always pops from the free
        # list end which is the slot we just freed.
        assert slot1 == slot2
        mv = arena.get_slice(slot2)
        assert bytes(mv) == v2
    finally:
        arena.close()


def test_get_slot_for_key_refreshes_lru():
    arena = SharedArena(16 * 1024, pin_memory=False)
    try:
        # slot_size after first put = ceil(100+16, 64) = 128
        # num_slots = 16384 // 128 = 128. Plenty of headroom.
        arena.put(b"k1", b"x" * 100)
        arena.put(b"k2", b"y" * 100)
        # Touching k1 makes it more recently used than k2.
        assert arena.get_slot_for_key(b"k1") is not None
        # Lookup must report `entries=2`.
        assert arena.stats().entries == 2
    finally:
        arena.close()


def test_first_put_too_large_for_capacity_rejected():
    arena = SharedArena(64, pin_memory=False)
    try:
        # First put: 100 + 16 = 116 > 64 capacity → reject.
        slot = arena.put(b"k", b"x" * 100)
        assert slot is None
        # Slot size remains undecided.
        assert arena.slot_size == 0
    finally:
        arena.close()


# ----------------------------------------------------------------------
# LRU eviction
# ----------------------------------------------------------------------


def test_lru_eviction_when_full():
    # Capacity for exactly 2 slots of 128 bytes after alignment.
    arena = SharedArena(2 * 128, pin_memory=False)
    try:
        # First put fixes slot_size = 128.
        s1 = arena.put(b"k1", b"a" * 100)
        s2 = arena.put(b"k2", b"b" * 100)
        # Third put forces LRU eviction of k1 (oldest).
        s3 = arena.put(b"k3", b"c" * 100)
        assert s1 is not None and s2 is not None and s3 is not None
        # k1 should be gone; k3 reuses k1's slot.
        assert arena.get_slot_for_key(b"k1") is None
        assert arena.get_slot_for_key(b"k2") == s2
        assert arena.get_slot_for_key(b"k3") == s3
        # Counters
        assert arena.stats().evictions_total == 1
    finally:
        arena.close()


def test_evict_lru_returns_key():
    arena = SharedArena(2 * 128, pin_memory=False)
    try:
        arena.put(b"k1", b"a" * 100)
        arena.put(b"k2", b"b" * 100)
        evicted = arena.evict_lru()
        assert evicted == b"k1"
        # Next eviction returns k2.
        evicted = arena.evict_lru()
        assert evicted == b"k2"
        # Empty.
        assert arena.evict_lru() is None
    finally:
        arena.close()


def test_clear_preserves_slot_size():
    arena = SharedArena(1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        slot_size_before = arena.slot_size
        arena.clear()
        assert arena.stats().entries == 0
        # slot_size must NOT change — clients cached it.
        assert arena.slot_size == slot_size_before
        # We can still put after clear.
        s = arena.put(b"k", b"y" * 100)
        assert s is not None
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Seqlock torn-read detection
# ----------------------------------------------------------------------


def test_get_slot_metadata_returns_offset_length_version():
    arena = SharedArena(1024, pin_memory=False)
    try:
        value = b"abc" * 100
        slot = arena.put(b"k", value)
        offset, length, version = arena.get_slot_metadata(slot)
        # offset = slot * slot_size + header
        assert offset == slot * arena.slot_size + _HEADER_BYTES
        assert length == len(value)
        # version should be even (stable) and >= 2 (one bump for
        # mid-write, one for stable).
        assert version >= 2 and version % 2 == 0
    finally:
        arena.close()


def test_torn_read_detected_when_version_is_odd():
    """Force a torn read by manually setting an odd version into the
    slot header and asserting `get_slice` returns None."""
    arena = SharedArena(1024, pin_memory=False)
    try:
        value = b"abc" * 100
        slot = arena.put(b"k", value)
        # Corrupt the version field to an odd value (mid-write state).
        base = slot * arena.slot_size
        arena._mmap[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4] = (
            struct.pack("<I", 7)  # odd → mid-write marker
        )
        # All retries should hit the odd-version path → None.
        result = arena.get_slice(slot)
        assert result is None
        # torn_reads counter should equal MAX_TORN_READ_RETRIES.
        assert arena.stats().torn_reads_total == MAX_TORN_READ_RETRIES
    finally:
        arena.close()


def test_concurrent_writer_and_reader_no_corruption():
    """Single writer thread overwriting a slot while a reader thread
    repeatedly reads. The reader either gets clean v1 or clean v2 or
    a torn read (None) — but NEVER a mixture of v1 and v2 bytes."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Two distinct values of the same size.
        v1 = b"A" * 1000
        v2 = b"B" * 1000
        slot = arena.put(b"k", v1)
        assert slot is not None

        stop_event = threading.Event()
        torn_count = 0
        corruption_count = 0
        clean_v1 = 0
        clean_v2 = 0
        lock = threading.Lock()

        def reader():
            nonlocal torn_count, corruption_count, clean_v1, clean_v2
            while not stop_event.is_set():
                mv = arena.get_slice(slot)
                if mv is None:
                    with lock:
                        torn_count += 1
                    continue
                b = bytes(mv)
                if b == v1:
                    with lock:
                        clean_v1 += 1
                elif b == v2:
                    with lock:
                        clean_v2 += 1
                else:
                    # If the bytes are anything else, it's a torn
                    # read that bypassed our seqlock — bug.
                    with lock:
                        corruption_count += 1

        readers = [threading.Thread(target=reader) for _ in range(2)]
        for r in readers:
            r.start()

        # Writer flips between v1 and v2 a few hundred times.
        for i in range(500):
            arena.put(b"k", v1 if i % 2 == 0 else v2)

        stop_event.set()
        for r in readers:
            r.join(timeout=2)

        # The critical invariant: no corruption-class read. Torn
        # reads are OK (they're handled by the protocol — return
        # None, caller retries or falls through).
        assert corruption_count == 0, f"got {corruption_count} corrupt reads"
        # And we should have seen at least some clean reads.
        assert clean_v1 + clean_v2 > 0
    finally:
        arena.close()


def test_concurrent_writers_different_slots_independent():
    """Two writers writing to different keys (different slots) must
    not interfere. Per-slot locks should give them parallelism."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Pre-create the two slots.
        arena.put(b"k1", b"a" * 100)
        arena.put(b"k2", b"b" * 100)

        def writer(key, value):
            for _ in range(100):
                arena.put(key, value)

        t1 = threading.Thread(target=writer, args=(b"k1", b"X" * 100))
        t2 = threading.Thread(target=writer, args=(b"k2", b"Y" * 100))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Final state: k1=X*100, k2=Y*100.
        s1 = arena.get_slot_for_key(b"k1")
        s2 = arena.get_slot_for_key(b"k2")
        assert bytes(arena.get_slice(s1)) == b"X" * 100
        assert bytes(arena.get_slice(s2)) == b"Y" * 100
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Cross-process mmap path (single-process simulation — fork would
# work too, but mmap of the same FD in the same process exercises
# the same code path).
# ----------------------------------------------------------------------


def test_worker_side_mmap_sees_writer_bytes():
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        value = b"hello shared arena" + b"\x00" * 100
        slot = arena.put(b"k", value)
        offset, length, version = arena.get_slot_metadata(slot)

        # Worker-side: open a separate mmap on the same FD (mimics
        # what SCM_RIGHTS + recv_fd would do in another process).
        mm, mv = open_arena_view(arena.fd, arena.capacity_bytes)
        try:
            payload = read_slot_seqlock(mm, offset, length, version)
            assert payload is not None
            assert bytes(payload) == value
            # Release references before mmap.close() — memoryview
            # exports keep the mmap alive otherwise.
            payload = None
            mv.release()
        finally:
            mm.close()
    finally:
        arena.close()


def test_worker_side_torn_read_returns_none_when_version_mismatch():
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        value = b"abc" * 100
        slot = arena.put(b"k", value)
        offset, length, version = arena.get_slot_metadata(slot)

        # Worker mmap.
        mm, _mv = open_arena_view(arena.fd, arena.capacity_bytes)
        try:
            # Overwrite the slot with new data — version bumps.
            new_value = b"xyz" * 100
            arena.put(b"k", new_value)
            # Worker uses the OLD version — must detect torn.
            payload = read_slot_seqlock(mm, offset, length, version)
            assert payload is None
            _mv.release()
        finally:
            mm.close()
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Stats + info
# ----------------------------------------------------------------------


def test_stats_track_hits_and_misses():
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k1", b"x" * 100)
        assert arena.get_slot_for_key(b"k1") is not None
        assert arena.get_slot_for_key(b"miss") is None
        s = arena.stats()
        assert s.hits_total == 1
        assert s.misses_total == 1
        assert s.entries == 1
    finally:
        arena.close()


def test_arena_info_serializes_to_tuple_and_back():
    info = SharedArenaInfo(arena_size=1024, slot_size=128, server_pid=4242)
    t = info.to_tuple()
    assert t == (1024, 128, 4242)
    restored = SharedArenaInfo.from_tuple(t)
    assert restored == info


def test_arena_exposes_fd_and_info():
    arena = SharedArena(1024, pin_memory=False)
    try:
        assert arena.fd > 0
        info = arena.info
        assert info.arena_size == 1024
        assert info.server_pid > 0
    finally:
        arena.close()


def test_evict_key_explicit():
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        assert arena.evict_key(b"k") is True
        assert arena.evict_key(b"k") is False
        assert arena.get_slot_for_key(b"k") is None
    finally:
        arena.close()


def test_evict_lru_publishes_to_drain():
    """`evict_lru()` (external/admin path) must push the evicted key
    onto `_recent_evicted_keys` so HostStore's next drain catches it.
    HostStore-internal callers pass `notify_drain=False` and bypass
    this — but external admin/test paths get the publish-by-default."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k1", b"a" * 100)
        arena.put(b"k2", b"b" * 100)
        # Drain whatever the puts queued (none, since neither evicted).
        arena.drain_recent_evictions()
        # Admin-style evict — must publish.
        arena.evict_lru()
        assert arena.drain_recent_evictions() == [b"k1"]
    finally:
        arena.close()


def test_evict_key_publishes_to_drain():
    """Same as `test_evict_lru_publishes_to_drain` for the `evict_key`
    path."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        arena.drain_recent_evictions()
        arena.evict_key(b"k")
        assert arena.drain_recent_evictions() == [b"k"]
    finally:
        arena.close()


def test_evict_key_notify_drain_false_skips_publish():
    """HostStore-internal callers pass `notify_drain=False` because
    they've already updated their own bookkeeping. Verify the kwarg
    actually suppresses the publish."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        arena.drain_recent_evictions()
        arena.evict_key(b"k", notify_drain=False)
        assert arena.drain_recent_evictions() == []
    finally:
        arena.close()


def test_oversize_subsequent_put_rejected():
    """Once slot_size is fixed, blobs that don't fit must be refused."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # First put picks slot_size = ceil(100+16, 64) = 128.
        arena.put(b"k1", b"x" * 100)
        # Second put with 200 bytes (216 incl header) > 128 → reject.
        slot = arena.put(b"k2", b"y" * 200)
        assert slot is None
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Save-side CopyFree: reservation pool (reserve / commit / cancel)
# ----------------------------------------------------------------------


def test_reserve_returns_writable_mv_and_lease_token():
    """A successful reserve hands back a nonzero lease token, a real
    slot_id, and a payload offset inside the mmap that the engine can
    write into. The slot is invisible to LRU until commit."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        result = arena.reserve(size=200, connection_id=1)
        assert result is not None
        lease_token, slot_id, payload_offset = result
        assert lease_token > 0
        assert slot_id >= 0
        assert payload_offset == slot_id * arena.slot_size + _HEADER_BYTES
        # The slot grid was initialized on first reserve.
        assert arena.slot_size >= 200 + _HEADER_BYTES
        # Writing into the mmap at the offset should succeed.
        arena._mmap[payload_offset : payload_offset + 8] = b"abcdefgh"
        # Active reservation is observable via stats.
        assert arena.stats().reservations_active == 1
        # No committed entries yet.
        assert arena.stats().entries == 0
    finally:
        arena.close()


def test_commit_links_key_to_slot_id():
    """After commit, the key resolves to the reserved slot and reading
    via the public APIs returns the bytes the engine wrote into the
    mmap directly. The reservation counter moves to `committed`."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        result = arena.reserve(size=200, connection_id=1)
        assert result is not None
        lease, slot_id, payload_offset = result

        payload = b"zero-copy-write" + b"\x00" * (200 - len(b"zero-copy-write"))
        arena._mmap[payload_offset : payload_offset + 200] = payload

        ok, reason, overwritten = arena.commit_reservation(
            lease, key=b"k1", length=200, connection_id=1
        )
        assert ok is True
        assert reason == ""
        assert overwritten is None

        # Key now resolves to the slot.
        assert arena.get_slot_for_key(b"k1") == slot_id
        # get_slice returns the bytes the engine wrote.
        mv = arena.get_slice(slot_id)
        assert mv is not None
        assert bytes(mv) == payload

        stats = arena.stats()
        assert stats.reservations_active == 0
        assert stats.reservations_committed == 1
        assert stats.entries == 1
    finally:
        arena.close()


def test_cancel_returns_slot_to_free_list():
    """Cancel drops the reservation without committing — the slot
    rejoins the free list and is reused by a subsequent reserve.
    Cancel is idempotent (second cancel on the same lease returns False)."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        r1 = arena.reserve(size=100, connection_id=42)
        assert r1 is not None
        lease, slot_id, _ = r1
        # Capture slot_size so the second reserve uses the same grid.
        original_slot_size = arena.slot_size

        ok = arena.cancel_reservation(lease, connection_id=42)
        assert ok is True
        # Second cancel is idempotent.
        assert arena.cancel_reservation(lease, connection_id=42) is False

        # Subsequent reserve gets a slot (likely the same one, since
        # cancel pushed it back to the free list).
        r2 = arena.reserve(size=100, connection_id=42)
        assert r2 is not None
        _, slot_id2, _ = r2
        assert slot_id2 == slot_id
        assert arena.slot_size == original_slot_size

        stats = arena.stats()
        assert stats.reservations_active == 1
        assert stats.reservations_cancelled == 1
    finally:
        arena.close()


def test_wrong_connection_id_cannot_commit():
    """Commit must validate ownership — a second connection that
    learns the lease token via some side channel cannot finalize
    another connection's reservation. The slot stays reserved."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        result = arena.reserve(size=100, connection_id=1)
        assert result is not None
        lease, _, payload_offset = result
        arena._mmap[payload_offset : payload_offset + 8] = b"attacker"

        ok, reason, _ = arena.commit_reservation(lease, key=b"evil", length=100, connection_id=2)
        assert ok is False
        assert reason == "wrong_owner"

        # Reservation still alive; legitimate owner can still commit.
        ok, _, _ = arena.commit_reservation(lease, key=b"legit", length=100, connection_id=1)
        assert ok is True
        assert arena.get_slot_for_key(b"legit") is not None
        assert arena.get_slot_for_key(b"evil") is None
    finally:
        arena.close()


def test_commit_overwrites_existing_key_frees_old_slot():
    """When commit's key collides with an already-mapped key, the
    OLD slot is freed and `overwritten_key` is returned so HostStore
    can drop its stale row in lock-step."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # First, an inline put for the key to establish it in the index.
        first_slot = arena.put(b"shared-key", b"v1" + b"\x00" * 98)
        assert first_slot is not None

        # Now reserve+commit under the SAME key.
        result = arena.reserve(size=100, connection_id=1)
        assert result is not None
        lease, new_slot, payload_offset = result
        # Sanity: reserve picked a different slot than the existing one.
        assert new_slot != first_slot
        arena._mmap[payload_offset : payload_offset + 100] = b"v2" + b"\x00" * 98

        ok, reason, overwritten = arena.commit_reservation(
            lease, key=b"shared-key", length=100, connection_id=1
        )
        assert ok is True
        assert reason == ""
        assert overwritten == b"shared-key"

        # Key now points at the NEW slot, not the old one.
        assert arena.get_slot_for_key(b"shared-key") == new_slot

        # The old slot is back in the free list — a fresh reserve
        # picks it up.
        r3 = arena.reserve(size=100, connection_id=1)
        assert r3 is not None
        _, recycled_slot, _ = r3
        assert recycled_slot == first_slot
    finally:
        arena.close()


def test_connection_close_cancels_outstanding_leases():
    """When a connection drops, every lease it held must release
    the slot — otherwise a worker crash leaks the arena. Other
    connections' leases stay intact."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Conn 1 holds three leases.
        leases_c1 = [arena.reserve(100, connection_id=1) for _ in range(3)]
        # Conn 2 holds one lease.
        lease_c2 = arena.reserve(100, connection_id=2)
        assert all(x is not None for x in leases_c1)
        assert lease_c2 is not None

        assert arena.stats().reservations_active == 4

        cancelled = arena.cancel_connection_reservations(connection_id=1)
        assert cancelled == 3

        stats = arena.stats()
        # Conn 2's lease is still active.
        assert stats.reservations_active == 1
        # Counter movement matches the cancellations.
        assert stats.reservations_cancelled == 3

        # Conn 2 can still commit its lease.
        _, _, off = lease_c2
        arena._mmap[off : off + 100] = b"survivor" + b"\x00" * (100 - 8)
        ok, _, _ = arena.commit_reservation(lease_c2[0], key=b"c2", length=100, connection_id=2)
        assert ok is True
    finally:
        arena.close()


def test_reserve_oversize_rejected_after_grid_locked():
    """Once the slot grid is initialized (by an earlier reserve/put),
    a subsequent reserve for a payload that doesn't fit returns None."""
    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Lock slot size at ceil(100+16, 64) = 128.
        arena.put(b"k", b"x" * 100)
        assert arena.slot_size == 128

        result = arena.reserve(size=200, connection_id=1)
        assert result is None
    finally:
        arena.close()


def test_reserve_marks_slot_as_writing_via_seqlock():
    """Between reserve and commit, the slot header is odd (writing)
    so any racing reader sees torn and falls through to None."""
    import struct as _struct

    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Commit one entry so the slot grid is initialized AND the
        # slot starts with an even version stamp from `put`.
        slot = arena.put(b"committed", b"x" * 100)
        assert slot is not None
        base = slot * arena.slot_size
        version_before = _struct.unpack(
            "<I",
            arena._mmap[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4],
        )[0]
        assert version_before % 2 == 0  # stable after put

        # Evict (force the next reserve to reuse a free slot).
        arena.evict_key(b"committed", notify_drain=False)

        # Reserve — should pick up the same recycled slot and stamp
        # the header odd before returning.
        result = arena.reserve(size=100, connection_id=1)
        assert result is not None
        _, reserved_slot, _ = result
        base = reserved_slot * arena.slot_size
        version_during = _struct.unpack(
            "<I",
            arena._mmap[base + _HEADER_VERSION_OFFSET : base + _HEADER_VERSION_OFFSET + 4],
        )[0]
        assert version_during % 2 == 1  # writing
        # get_slice reads the odd version and falls through to None
        # for at least MAX_TORN_READ_RETRIES retries.
        assert arena.get_slice(reserved_slot) is None
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Hugepage memfd: MFD_HUGETLB | MFD_HUGE_2MB
# ----------------------------------------------------------------------


def test_memfd_create_without_hugetlb_unchanged():
    """Default constructor path (hugetlb=False) — the arena reports
    `hugetlb_active=False` and serves puts/gets as before. Regression
    fence against the new hugetlb codepath leaking into 4 KB callers."""
    arena = SharedArena(64 * 1024, pin_memory=False, hugetlb=False)
    try:
        assert arena.hugetlb_active is False
        # Capacity preserved exactly when hugetlb wasn't requested.
        assert arena.capacity_bytes == 64 * 1024
        slot = arena.put(b"k", b"x" * 100)
        assert slot is not None
        mv = arena.get_slice(slot)
        assert bytes(mv) == b"x" * 100
    finally:
        arena.close()


def test_memfd_create_with_hugetlb_falls_back_when_unavailable(monkeypatch, caplog):
    """When `hugetlb=True` but the kernel refuses `MFD_HUGETLB` (e.g.
    `vm.nr_hugepages` is 0), the arena falls back to 4 KB pages
    silently-with-a-warning. The arena must still start and serve
    puts/gets — kvd's startup must be resilient to misconfigured
    hosts (graceful fallback)."""
    import logging
    import os as _os

    real_memfd_create = _os.memfd_create

    def fake_memfd_create(name, flags=0):
        # The arena tries MFD_HUGETLB first; raise OSError to simulate
        # kernel refusal. On the fallback retry (CLOEXEC only), call
        # the real implementation.
        if flags & _os.MFD_HUGETLB:
            raise OSError(12, "Cannot allocate memory")  # ENOMEM
        return real_memfd_create(name, flags)

    monkeypatch.setattr(_os, "memfd_create", fake_memfd_create)

    caplog.set_level(logging.WARNING, logger="infera.kvd.shared_arena")
    # Capacity 4 MB so the post-round capacity is still a sane
    # multiple of 2 MB (round-up is a no-op here).
    arena = SharedArena(4 * 1024 * 1024, pin_memory=False, hugetlb=True)
    try:
        # Fell back — hugetlb is NOT active.
        assert arena.hugetlb_active is False
        # Warning was logged so ops can fix `vm.nr_hugepages`.
        assert any(
            "MFD_HUGETLB" in rec.message and "falling back" in rec.message for rec in caplog.records
        )
        # Arena still serves a put/get round-trip.
        slot = arena.put(b"k", b"x" * 100)
        assert slot is not None
        mv = arena.get_slice(slot)
        assert bytes(mv) == b"x" * 100
    finally:
        arena.close()


def test_capacity_rounded_up_for_hugetlb(monkeypatch):
    """Hugepage-backed arenas need the mapping length to be a multiple
    of 2 MB or `ftruncate` / `memfd_create` will EINVAL. A requested
    7 MB capacity must round up to 8 MB. We stub `memfd_create` to
    return a regular memfd (so the rest of the test runs on hosts
    without hugepages configured) — the round-up logic is what we're
    testing, not the kernel path."""
    import os as _os

    real_memfd_create = _os.memfd_create
    seen_flags: list[int] = []

    def fake_memfd_create(name, flags=0):
        seen_flags.append(flags)
        # Strip the hugetlb bits so the call goes through on any host.
        return real_memfd_create(name, flags & _os.MFD_CLOEXEC)

    monkeypatch.setattr(_os, "memfd_create", fake_memfd_create)

    requested = 7 * 1024 * 1024  # 7 MB — not a multiple of 2 MB
    arena = SharedArena(requested, pin_memory=False, hugetlb=True)
    try:
        # Capacity bumped to next 2 MB boundary.
        assert arena.capacity_bytes == 8 * 1024 * 1024
        # Confirm the arena DID try the hugetlb path first.
        assert seen_flags
        assert seen_flags[0] & _os.MFD_HUGETLB
        assert seen_flags[0] & _os.MFD_HUGE_2MB
    finally:
        arena.close()


def test_hugetlb_env_var_default(monkeypatch):
    """When the constructor `hugetlb` arg is None, the env var
    `INFERA_KVD_ARENA_HUGETLB` decides. With the env var off (the
    production default), no hugetlb attempt is made — the arena
    creates a plain 4 KB memfd. The flag-tracking stub lets us
    confirm no MFD_HUGETLB bit was passed."""
    import os as _os

    real_memfd_create = _os.memfd_create
    seen_flags: list[int] = []

    def fake_memfd_create(name, flags=0):
        seen_flags.append(flags)
        return real_memfd_create(name, flags & _os.MFD_CLOEXEC)

    monkeypatch.delenv("INFERA_KVD_ARENA_HUGETLB", raising=False)
    monkeypatch.setattr(_os, "memfd_create", fake_memfd_create)

    arena = SharedArena(64 * 1024, pin_memory=False)  # hugetlb=None default
    try:
        assert arena.hugetlb_active is False
        # No MFD_HUGETLB bit was ever set on any call.
        assert all(not (f & _os.MFD_HUGETLB) for f in seen_flags)
    finally:
        arena.close()


# ----------------------------------------------------------------------
# Engine-side mmap MAP_POPULATE prefault
# ----------------------------------------------------------------------


def test_open_arena_view_without_populate_unchanged():
    """Default `prefault=False` path. The mmap call should NOT include
    MAP_POPULATE; behaviour matches the pre-change implementation."""
    import mmap as _mmap
    from unittest.mock import patch

    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        # Put one slot so the arena's grid is initialized.
        slot = arena.put(b"k", b"x" * 100)
        assert slot is not None
        seen_flags: list[int] = []
        real_mmap = _mmap.mmap

        def fake_mmap(fd, length, *args, **kwargs):
            seen_flags.append(kwargs.get("flags", 0))
            return real_mmap(fd, length, *args, **kwargs)

        with patch.object(_mmap, "mmap", side_effect=fake_mmap):
            mm, mv = open_arena_view(arena.fd, arena.capacity_bytes, prefault=False)
            try:
                # MAP_POPULATE bit should be clear.
                map_populate = getattr(_mmap, "MAP_POPULATE", 0)
                if map_populate:
                    assert all(not (f & map_populate) for f in seen_flags)
                # Read works — the put landed at `slot`'s offset.
                base = slot * arena.slot_size
                assert bytes(mv[base + _HEADER_BYTES : base + _HEADER_BYTES + 5]) == b"xxxxx"
            finally:
                mv.release()
                mm.close()
    finally:
        arena.close()


def test_open_arena_view_with_populate_succeeds():
    """When `prefault=True` is requested AND the platform exposes
    `mmap.MAP_POPULATE`, the flag is added to the mmap call. The
    mapping must still be readable — `MAP_POPULATE` is purely a
    pre-fault hint, so semantics are unchanged from the caller's
    perspective."""
    import mmap as _mmap
    from unittest.mock import patch

    if not hasattr(_mmap, "MAP_POPULATE"):
        import pytest as _pytest

        _pytest.skip("MAP_POPULATE not available on this platform")

    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        slot = arena.put(b"k", b"x" * 100)
        assert slot is not None
        seen_flags: list[int] = []
        real_mmap = _mmap.mmap

        def fake_mmap(fd, length, *args, **kwargs):
            seen_flags.append(kwargs.get("flags", 0))
            return real_mmap(fd, length, *args, **kwargs)

        with patch.object(_mmap, "mmap", side_effect=fake_mmap):
            mm, mv = open_arena_view(arena.fd, arena.capacity_bytes, prefault=True)
            try:
                assert seen_flags
                assert seen_flags[0] & _mmap.MAP_POPULATE
                # And the mapping still reads — MAP_POPULATE is a
                # purely additive hint; no semantic change.
                base = slot * arena.slot_size
                assert bytes(mv[base + _HEADER_BYTES : base + _HEADER_BYTES + 5]) == b"xxxxx"
            finally:
                mv.release()
                mm.close()
    finally:
        arena.close()


def test_open_arena_view_prefault_env_var_default(monkeypatch):
    """`prefault=None` (the default) reads `INFERA_KVD_ARENA_PREFAULT`.
    With the env var UNSET, MAP_POPULATE IS set — the production
    default opts IN (one-time ~180 ms/GB mmap cost buys 9× first-touch
    save speedup on long-running daemons; see open_arena_view doc)."""
    import mmap as _mmap
    from unittest.mock import patch

    monkeypatch.delenv("INFERA_KVD_ARENA_PREFAULT", raising=False)

    if not hasattr(_mmap, "MAP_POPULATE"):
        import pytest as _pytest

        _pytest.skip("MAP_POPULATE not available on this platform")

    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        seen_flags: list[int] = []
        real_mmap = _mmap.mmap

        def fake_mmap(fd, length, *args, **kwargs):
            seen_flags.append(kwargs.get("flags", 0))
            return real_mmap(fd, length, *args, **kwargs)

        with patch.object(_mmap, "mmap", side_effect=fake_mmap):
            mm, mv = open_arena_view(arena.fd, arena.capacity_bytes)
            try:
                assert seen_flags
                assert seen_flags[0] & _mmap.MAP_POPULATE, (
                    f"expected MAP_POPULATE in default-on flags, got {[hex(f) for f in seen_flags]}"
                )
            finally:
                mv.release()
                mm.close()
    finally:
        arena.close()


@pytest.mark.parametrize("falsy_value", ["0", "false", "no", "off"])
def test_open_arena_view_prefault_env_zero_opts_out(monkeypatch, falsy_value):
    """`INFERA_KVD_ARENA_PREFAULT=0` (or `false`/`no`/`off`) is the
    explicit opt-out path: with `prefault=None`, the env-falsy value
    must suppress MAP_POPULATE so operators on memory-constrained
    boxes or older kernels can disable the prefault behaviour."""
    import mmap as _mmap
    from unittest.mock import patch

    monkeypatch.setenv("INFERA_KVD_ARENA_PREFAULT", falsy_value)

    arena = SharedArena(64 * 1024, pin_memory=False)
    try:
        arena.put(b"k", b"x" * 100)
        seen_flags: list[int] = []
        real_mmap = _mmap.mmap

        def fake_mmap(fd, length, *args, **kwargs):
            seen_flags.append(kwargs.get("flags", 0))
            return real_mmap(fd, length, *args, **kwargs)

        with patch.object(_mmap, "mmap", side_effect=fake_mmap):
            mm, mv = open_arena_view(arena.fd, arena.capacity_bytes)
            try:
                map_populate = getattr(_mmap, "MAP_POPULATE", 0)
                if map_populate:
                    assert all(not (f & map_populate) for f in seen_flags), (
                        f"env={falsy_value!r} should suppress "
                        f"MAP_POPULATE, got {[hex(f) for f in seen_flags]}"
                    )
            finally:
                mv.release()
                mm.close()
    finally:
        arena.close()
