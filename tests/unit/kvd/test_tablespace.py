###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/tablespace.py.

Coverage:
  - BitsetAllocator (alloc/free/mark_used semantics, exhaustion, ranges)
  - TablespaceJournal (append, replay, truncate-tail-on-corruption)
  - TablespaceLongRegion put/get round-trip, eviction, restart-survival
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from infera.kvd.tablespace import (
    _FS_DEFAULTS,
    BitsetAllocator,
    TablespaceEntry,
    TablespaceJournal,
    TablespaceLongRegion,
    detect_fs_defaults,
)

# ----------------------------------------------------------------------
# BitsetAllocator
# ----------------------------------------------------------------------


def test_bitset_alloc_starts_empty():
    a = BitsetAllocator(10)
    assert a.total_slots == 10
    assert a.allocated == 0
    assert a.num_free == 10


def test_bitset_alloc_returns_consecutive_slots_initially():
    """The linear-scan allocator should hand out 0, 1, 2, ... on a fresh
    bitset — operators reading slot indices in logs benefit from this
    being predictable."""
    a = BitsetAllocator(8)
    slots = [a.alloc() for _ in range(5)]
    assert slots == [0, 1, 2, 3, 4]
    assert a.allocated == 5


def test_bitset_alloc_returns_none_when_full():
    a = BitsetAllocator(4)
    for _ in range(4):
        assert a.alloc() is not None
    assert a.alloc() is None
    assert a.num_free == 0


def test_bitset_free_frees_slot():
    a = BitsetAllocator(8)
    s0 = a.alloc()
    s1 = a.alloc()
    assert s0 == 0 and s1 == 1
    a.free(s0)
    assert a.allocated == 1
    # Next alloc reuses 0 (lowest free bit).
    assert a.alloc() == 0


def test_bitset_free_idempotent():
    """Free on an already-free slot should be a no-op (no underflow)."""
    a = BitsetAllocator(8)
    a.free(3)  # never allocated
    assert a.allocated == 0


def test_bitset_free_out_of_range_is_noop():
    a = BitsetAllocator(8)
    a.alloc()
    a.free(100)  # out of range
    a.free(-1)
    assert a.allocated == 1  # unchanged


def test_bitset_is_set_reports_state_correctly():
    a = BitsetAllocator(8)
    s = a.alloc()
    assert a.is_set(s)
    assert not a.is_set(s + 1)
    a.free(s)
    assert not a.is_set(s)


def test_bitset_mark_used_for_restart_replay():
    """Used by restart code to assert that a slot is allocated without
    re-running the linear scan. Returns True on first mark, False on dup."""
    a = BitsetAllocator(8)
    assert a.mark_used(5) is True
    assert a.allocated == 1
    assert a.is_set(5)
    # Duplicate mark: returns False, no double-count.
    assert a.mark_used(5) is False
    assert a.allocated == 1


def test_bitset_mark_used_out_of_range_raises():
    a = BitsetAllocator(8)
    with pytest.raises(ValueError):
        a.mark_used(100)
    with pytest.raises(ValueError):
        a.mark_used(-1)


def test_bitset_full_then_free_then_alloc_round_trip():
    """Stress: fill, free a few middle slots, reallocate."""
    a = BitsetAllocator(64)
    slots = [a.alloc() for _ in range(64)]
    assert all(s is not None for s in slots)
    assert a.num_free == 0
    a.free(10)
    a.free(20)
    a.free(30)
    assert a.num_free == 3
    new = [a.alloc(), a.alloc(), a.alloc()]
    assert sorted(new) == [10, 20, 30]
    assert a.alloc() is None  # full again


def test_bitset_handles_non_byte_aligned_total_slots():
    """13 slots fits in 2 bytes but only the first 13 bits are valid;
    alloc must not hand out slots 13-15."""
    a = BitsetAllocator(13)
    slots = []
    for _ in range(15):
        s = a.alloc()
        if s is None:
            break
        slots.append(s)
    assert slots == list(range(13))
    assert a.alloc() is None


# ----------------------------------------------------------------------
# TablespaceJournal
# ----------------------------------------------------------------------


def test_journal_append_and_read(tmp_path: Path):
    j = TablespaceJournal(tmp_path / "j.log", sync_writes=False)
    j.open()
    try:
        j.append({"op": "PUT", "key_hex": "ab"})
        j.append({"op": "DEL", "key_hex": "cd"})
    finally:
        j.close()

    entries = TablespaceJournal(tmp_path / "j.log").read_all()
    assert entries == [
        {"op": "PUT", "key_hex": "ab"},
        {"op": "DEL", "key_hex": "cd"},
    ]


def test_journal_read_all_returns_empty_when_file_missing(tmp_path: Path):
    j = TablespaceJournal(tmp_path / "nonexistent.log")
    assert j.read_all() == []


def test_journal_skips_truncated_last_line(tmp_path: Path):
    """If the daemon crashed mid-write the last line may be truncated.
    read_all must drop it and return everything before."""
    p = tmp_path / "j.log"
    p.write_bytes(
        b'{"op":"PUT","key_hex":"aa"}\n{"op":"PUT","key_hex":"bb"}\n{"op":"PUT","key_h'  # truncated
    )
    entries = TablespaceJournal(p).read_all()
    assert len(entries) == 2
    assert entries[0]["key_hex"] == "aa"
    assert entries[1]["key_hex"] == "bb"


def test_journal_skips_nul_padded_tail(tmp_path: Path):
    """Some filesystems pad the tail with NUL bytes on crash. The
    `for line in f` iterator + strip() will skip past the NUL run
    looking for a real newline; verify we read the valid entries
    and don't choke. Originally only explicit truncation was
    exercised; this also catches NUL-padded corruption."""
    p = tmp_path / "j.log"
    p.write_bytes(
        b'{"op":"PUT","key_hex":"aa"}\n'
        b'{"op":"PUT","key_hex":"bb"}\n'
        b"\x00" * 200  # NUL-padded tail (simulates crash on some FSes)
    )
    entries = TablespaceJournal(p).read_all()
    # Should read at least the two valid entries; the NUL tail is
    # treated as junk lines and dropped.
    valid_keys = [e["key_hex"] for e in entries if "key_hex" in e]
    assert "aa" in valid_keys
    assert "bb" in valid_keys


def test_journal_skips_middle_corruption(tmp_path: Path):
    """A line with garbage in the middle of the file (e.g., a partial
    write that didn't fail cleanly) should be dropped — entries
    before it are valid; we lose tail visibility but don't crash."""
    p = tmp_path / "j.log"
    p.write_bytes(
        b'{"op":"PUT","key_hex":"aa"}\n'
        b"GARBAGE-NOT-JSON\n"  # mid-stream corruption
        b'{"op":"PUT","key_hex":"cc"}\n'  # would be valid but after corruption
    )
    entries = TablespaceJournal(p).read_all()
    # First entry must come through; corruption truncates the rest.
    assert len(entries) >= 1
    assert entries[0]["key_hex"] == "aa"


def test_journal_skips_partial_first_line(tmp_path: Path):
    """A journal with ONLY a partial line (crash on first write) must
    return [] rather than raise or returning garbage."""
    p = tmp_path / "j.log"
    p.write_bytes(b'{"op":"PUT","key_h')  # crashed mid-first-write
    entries = TablespaceJournal(p).read_all()
    assert entries == []


def test_journal_truncate_drops_existing_entries(tmp_path: Path):
    j = TablespaceJournal(tmp_path / "j.log", sync_writes=False)
    j.open()
    j.append({"op": "PUT"})
    j.append({"op": "PUT"})
    j.truncate()
    # File is now empty (but exists for further appends).
    assert TablespaceJournal(tmp_path / "j.log").read_all() == []
    # Can still append after truncate.
    j.append({"op": "PUT", "after": "truncate"})
    j.close()
    entries = TablespaceJournal(tmp_path / "j.log").read_all()
    assert entries == [{"op": "PUT", "after": "truncate"}]


# ----------------------------------------------------------------------
# TablespaceLongRegion — basic round-trip
# ----------------------------------------------------------------------


def _make_region(tmp_path: Path, **kw) -> TablespaceLongRegion:
    """Default tiny region for unit tests. 8 KB slots × 4 slots per
    container × 4 containers = 128 KB total, 16 slots.

    Default `o_direct=False` here because most tests use 8 KB slots
    (smaller than the 4 KB O_DIRECT alignment unit, so they'd just
    add noise — those tests aren't about IO mode). O_DIRECT behavior
    is exercised separately in the `_o_direct_*` tests with
    correctly-aligned slots."""
    return TablespaceLongRegion(
        path=tmp_path,
        max_bytes=kw.pop("max_bytes", 128 * 1024),
        slot_bytes=kw.pop("slot_bytes", 8 * 1024),
        container_bytes=kw.pop("container_bytes", 32 * 1024),
        sync_writes=kw.pop("sync_writes", False),
        o_direct=kw.pop("o_direct", False),
    )


def test_put_get_round_trip(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        payload = b"hello, world" + b"\x00" * 100
        accepted, reason = r.put(b"k1", payload, retention="long", model="m", compat_key="ck")
        assert accepted, reason
        got = r.get_bytes(b"k1", model="m", compat_key="ck")
        assert got == payload
    finally:
        r.shutdown()


def test_put_returns_false_for_short_retention(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, reason = r.put(b"k", b"v", retention="short", model="m", compat_key="ck")
        assert ok is False
        assert "only_accepts_long_retention" in (reason or "")
    finally:
        r.shutdown()


def test_put_rejects_oversized_value(tmp_path: Path):
    r = _make_region(tmp_path, slot_bytes=64)
    r.start()
    try:
        ok, reason = r.put(b"k", b"x" * 65, retention="long", model="m", compat_key="ck")
        assert ok is False
        assert "exceeds_slot_bytes" in (reason or "")
    finally:
        r.shutdown()


def test_put_rejects_empty_value(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, reason = r.put(b"k", b"", retention="long", model="m", compat_key="ck")
        assert ok is False
        assert "empty_value" in (reason or "")
    finally:
        r.shutdown()


def test_get_bytes_returns_none_on_corrupt_size_zero(tmp_path: Path):
    """Regression: if a malformed/corrupted
    index entry sets size_bytes=0, _aligned_read would return zero
    bytes silently, masking the corruption. GET must instead miss with
    a warning."""
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, _ = r.put(b"k", b"the payload", retention="long", model="m", compat_key="ck")
        assert ok
        # Corrupt the entry's size_bytes to simulate a bad replay.
        entry = r.get_entry(b"k", model="m", compat_key="ck")
        assert entry is not None
        entry.size_bytes = 0
        assert r.get_bytes(b"k", model="m", compat_key="ck") is None
    finally:
        r.shutdown()


def test_get_bytes_returns_none_on_corrupt_size_oversized(tmp_path: Path):
    """Same as above but for size_bytes > slot_bytes — the slot can't
    physically hold that, so GET must miss rather than read out of
    range (and risk crossing into the next slot's bytes)."""
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, _ = r.put(b"k", b"the payload", retention="long", model="m", compat_key="ck")
        assert ok
        entry = r.get_entry(b"k", model="m", compat_key="ck")
        assert entry is not None
        entry.size_bytes = r._slot_bytes + 1
        assert r.get_bytes(b"k", model="m", compat_key="ck") is None
    finally:
        r.shutdown()


def test_put_existing_key_updates_in_place(tmp_path: Path):
    """A second PUT for the same key reuses the same slot."""
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k", b"first", retention="long", model="m", compat_key="ck")
        entry_v1 = r.get_entry(b"k", model="m", compat_key="ck")
        assert entry_v1 is not None
        slot_v1 = (entry_v1.container_idx, entry_v1.slot_idx)

        r.put(b"k", b"second value!", retention="long", model="m", compat_key="ck")
        entry_v2 = r.get_entry(b"k", model="m", compat_key="ck")
        slot_v2 = (entry_v2.container_idx, entry_v2.slot_idx)

        # Same slot location.
        assert slot_v1 == slot_v2
        # Updated bytes.
        assert r.get_bytes(b"k", model="m", compat_key="ck") == b"second value!"
        # Used bytes reflects new size, not old + new.
        assert r.used_bytes == len(b"second value!")
    finally:
        r.shutdown()


def test_get_miss_returns_none(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        assert r.get_bytes(b"absent") is None
    finally:
        r.shutdown()


def test_exists_returns_per_key_booleans(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"a", b"x", retention="long", model="m", compat_key="ck")
        r.put(b"b", b"y", retention="long", model="m", compat_key="ck")
        result = r.exists([b"a", b"missing", b"b"], model="m", compat_key="ck")
        assert result == [True, False, True]
    finally:
        r.shutdown()


def test_namespace_isolation_via_compat_key(tmp_path: Path):
    """Two TP ranks of the same model use distinct compat_keys → distinct
    slots; one rank's bytes never override the other's even if the
    `key` bytes collide."""
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k", b"tp0-bytes", retention="long", model="m", compat_key="tp0of2")
        r.put(b"k", b"tp1-bytes", retention="long", model="m", compat_key="tp1of2")

        # Both readable independently.
        assert r.get_bytes(b"k", model="m", compat_key="tp0of2") == b"tp0-bytes"
        assert r.get_bytes(b"k", model="m", compat_key="tp1of2") == b"tp1-bytes"
        # Index has two distinct entries.
        assert r.entries_count == 2
    finally:
        r.shutdown()


def test_remove_drops_entry(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k", b"v", retention="long", model="m", compat_key="ck")
        removed = r.remove(b"k", model="m", compat_key="ck")
        assert removed is True
        assert r.get_bytes(b"k", model="m", compat_key="ck") is None
        removed_again = r.remove(b"k", model="m", compat_key="ck")
        assert removed_again is False  # already gone
    finally:
        r.shutdown()


def test_clear_drops_everything(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        for i in range(5):
            r.put(f"k{i}".encode(), b"value", retention="long", model="m", compat_key="ck")
        n = r.clear()
        assert n == 5
        assert r.entries_count == 0
        assert r.used_bytes == 0
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Eviction (LRU)
# ----------------------------------------------------------------------


def test_eviction_when_slots_exhausted(tmp_path: Path):
    """Fill all slots, then PUT one more → LRU victim evicted."""
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=16 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r.start()
    try:
        # 64K / 8K = 8 slots total. Fill them.
        for i in range(8):
            ok, _ = r.put(f"k{i}".encode(), b"v" * 100, retention="long", model="m", compat_key="c")
            assert ok
        assert r.entries_count == 8

        # Touch k1 so it's MRU; touch k0 LRU implicitly stays oldest.
        r.get_bytes(b"k1", model="m", compat_key="c")

        # Insert k8 — should evict k0 (oldest by last_access).
        ok, _ = r.put(b"k8", b"v" * 100, retention="long", model="m", compat_key="c")
        assert ok
        assert r.entries_count == 8  # still 8, one was evicted
        assert r.get_bytes(b"k0", model="m", compat_key="c") is None
        assert r.get_bytes(b"k8", model="m", compat_key="c") == b"v" * 100
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Restart survival
# ----------------------------------------------------------------------


def test_restart_recovers_index_via_journal(tmp_path: Path):
    """No graceful shutdown → no snapshot. The journal alone must let
    a fresh region come back up with the same blocks."""
    payload = b"persistent payload " + b"\xaa" * 100

    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"k", payload, retention="long", model="m", compat_key="ck")
    # NB: NO shutdown(). Simulate a crash by just dropping the reference.
    # The container fds will be closed when r1 is garbage-collected, but
    # the journal entry is already fsynced.
    del r1

    r2 = _make_region(tmp_path)
    r2.start()
    try:
        assert r2.entries_count == 1
        got = r2.get_bytes(b"k", model="m", compat_key="ck")
        assert got == payload
    finally:
        r2.shutdown()


def test_graceful_shutdown_writes_snapshot_and_truncates_journal(tmp_path: Path):
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"k1", b"a", retention="long", model="m", compat_key="ck")
    r1.put(b"k2", b"b", retention="long", model="m", compat_key="ck")
    r1.shutdown()

    # Snapshot exists; journal is empty.
    assert (tmp_path / "index.snapshot.json").exists()
    j = TablespaceJournal(tmp_path / "index.log")
    assert j.read_all() == []

    r2 = _make_region(tmp_path)
    r2.start()
    try:
        assert r2.entries_count == 2
        assert r2.get_bytes(b"k1", model="m", compat_key="ck") == b"a"
        assert r2.get_bytes(b"k2", model="m", compat_key="ck") == b"b"
    finally:
        r2.shutdown()


def test_snapshot_plus_journal_replay(tmp_path: Path):
    """Snapshot reflects state at last shutdown; journal entries since
    the snapshot are layered on top."""
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"old", b"x", retention="long", model="m", compat_key="c")
    r1.shutdown()  # snapshot reflects {old: x}

    r2 = _make_region(tmp_path)
    r2.start()
    # After this PUT, snapshot still says {old: x}, journal says +new.
    r2.put(b"new", b"y", retention="long", model="m", compat_key="c")
    # Simulate crash (no shutdown).
    del r2

    r3 = _make_region(tmp_path)
    r3.start()
    try:
        assert r3.entries_count == 2
        assert r3.get_bytes(b"old", model="m", compat_key="c") == b"x"
        assert r3.get_bytes(b"new", model="m", compat_key="c") == b"y"
    finally:
        r3.shutdown()


def test_journal_del_replay_removes_entry(tmp_path: Path):
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"k", b"v", retention="long", model="m", compat_key="c")
    r1.shutdown()  # snapshot {k: v}

    r2 = _make_region(tmp_path)
    r2.start()
    r2.remove(b"k", model="m", compat_key="c")  # journal: DEL k
    del r2  # crash without snapshot

    r3 = _make_region(tmp_path)
    r3.start()
    try:
        assert r3.entries_count == 0
        assert r3.get_bytes(b"k", model="m", compat_key="c") is None
    finally:
        r3.shutdown()


def test_snapshot_corrupt_falls_back_to_bak(tmp_path: Path):
    """If primary snapshot is corrupt, .bak must be tried."""
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"first", b"a", retention="long", model="m", compat_key="c")
    r1.shutdown()  # snapshot v1 written

    r2 = _make_region(tmp_path)
    r2.start()
    r2.put(b"second", b"b", retention="long", model="m", compat_key="c")
    r2.shutdown()  # snapshot v2 written; v1 rotated to .bak

    # Corrupt the primary snapshot. .bak still holds v1.
    (tmp_path / "index.snapshot.json").write_bytes(b"junk{not-json")

    r3 = _make_region(tmp_path)
    r3.start()
    try:
        # Falls back to .bak which only had 'first'.
        assert r3.entries_count == 1
        assert r3.get_bytes(b"first", model="m", compat_key="c") == b"a"
        # 'second' lived only in the (corrupted) primary; lost on fallback.
        # In practice the journal would still have the PUT, but
        # snapshot+journal compaction makes this a real edge case we
        # accept the loss for. Document this as a known limitation.
        assert r3.get_bytes(b"second", model="m", compat_key="c") is None
    finally:
        r3.shutdown()


def test_snapshot_geometry_mismatch_falls_back_to_journal(tmp_path: Path):
    """Operator changes slot_bytes between runs → snapshot's slot/container
    indices are no longer valid for the new geometry. We must IGNORE
    the snapshot rather than mis-decode."""
    r1 = TablespaceLongRegion(
        tmp_path,
        max_bytes=16 * 1024,
        slot_bytes=512,
        container_bytes=4 * 1024,
        sync_writes=False,
        o_direct=False,  # 512-byte slot is sub-4K → can't use O_DIRECT
    )
    r1.start()
    r1.put(b"k", b"v", retention="long", model="m", compat_key="c")
    r1.shutdown()

    # Now restart with a DIFFERENT slot size.
    r2 = TablespaceLongRegion(
        tmp_path,
        max_bytes=16 * 1024,
        slot_bytes=1024,
        container_bytes=4 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r2.start()
    try:
        # Snapshot is geometry-mismatched → ignored. Journal is empty
        # (we did a graceful shutdown). So the region is empty.
        # The bytes are still on disk in the container files but they're
        # unreachable.
        assert r2.entries_count == 0
    finally:
        r2.shutdown()


# ----------------------------------------------------------------------
# Container preallocation
# ----------------------------------------------------------------------


def test_container_files_preallocated_to_full_size(tmp_path: Path):
    """fallocate (or ftruncate fallback) must ensure the container is
    full-sized after `start()` — slot writes assume the byte offset
    is reachable."""
    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=16 * 1024,
        slot_bytes=1024,
        container_bytes=4 * 1024,
        o_direct=False,  # sub-4K slot
    )
    r.start()
    try:
        for i in range(4):
            cp = tmp_path / "containers" / f"{i:04d}.bin"
            assert cp.exists()
            assert cp.stat().st_size == 4 * 1024
    finally:
        r.shutdown()


def test_existing_container_files_reopened(tmp_path: Path):
    """Second start() must not re-create or truncate containers."""
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"k", b"some value", retention="long", model="m", compat_key="c")
    r1.shutdown()

    # Note the file mtime/inode before restart.
    container_path = tmp_path / "containers" / "0000.bin"
    inode_before = container_path.stat().st_ino

    r2 = _make_region(tmp_path)
    r2.start()
    try:
        # Same inode (file was reused, not recreated).
        assert container_path.stat().st_ino == inode_before
        # Bytes still readable.
        assert r2.get_bytes(b"k", model="m", compat_key="c") == b"some value"
    finally:
        r2.shutdown()


# ----------------------------------------------------------------------
# Stats + properties
# ----------------------------------------------------------------------


def test_used_bytes_tracks_size_not_slot_size(tmp_path: Path):
    """used_bytes reports actual value bytes, NOT slot bytes — the
    slot might be 64 KB but the value 5 KB → used_bytes += 5K."""
    r = _make_region(tmp_path, slot_bytes=8 * 1024)
    r.start()
    try:
        r.put(b"k", b"x" * 100, retention="long", model="m", compat_key="c")
        assert r.used_bytes == 100
    finally:
        r.shutdown()


def test_properties_are_immutable_after_init(tmp_path: Path):
    r = _make_region(tmp_path, slot_bytes=8 * 1024, container_bytes=32 * 1024)
    assert r.slot_bytes == 8 * 1024
    assert r.container_bytes == 32 * 1024
    assert r.num_containers == 4  # 128K / 32K
    assert r.max_bytes == 128 * 1024


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_max_bytes_smaller_than_container_uses_one_container(tmp_path: Path):
    """Tiny max_bytes shouldn't crash; it just uses a single container
    sized to whatever the operator asked for."""
    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=2048,
        slot_bytes=512,  # sub-4K slot incompatible with O_DIRECT
        container_bytes=1 << 20,
        sync_writes=False,
        o_direct=False,
    )
    # container_bytes > max_bytes is OK — we just won't fill it.
    assert r.num_containers == 1


def test_invalid_construction():
    # Each of these triggers a ValueError BEFORE the o_direct alignment
    # check, so we don't need to opt out of O_DIRECT here.
    with pytest.raises(ValueError):
        TablespaceLongRegion("/tmp/x", max_bytes=0, slot_bytes=64, container_bytes=128)
    with pytest.raises(ValueError):
        TablespaceLongRegion("/tmp/x", max_bytes=1024, slot_bytes=0, container_bytes=128)
    with pytest.raises(ValueError):
        # container smaller than slot
        TablespaceLongRegion("/tmp/x", max_bytes=1024, slot_bytes=128, container_bytes=64)
    # New: default is o_direct=True, so non-4K-aligned slot bytes are
    # now invalid by default (the alignment check fires).
    with pytest.raises(ValueError, match="multiple of 4096"):
        TablespaceLongRegion("/tmp/x", max_bytes=8192, slot_bytes=1024, container_bytes=4096)


def test_double_start_is_idempotent(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    r.start()  # second call should be a no-op
    try:
        r.put(b"k", b"v", retention="long", model="m", compat_key="c")
        assert r.get_bytes(b"k", model="m", compat_key="c") == b"v"
    finally:
        r.shutdown()


def test_shutdown_before_start_is_safe(tmp_path: Path):
    r = _make_region(tmp_path)
    # Should not raise.
    r.shutdown()


def test_put_before_start_returns_error(tmp_path: Path):
    r = _make_region(tmp_path)
    ok, reason = r.put(b"k", b"v", retention="long", model="m", compat_key="c")
    assert ok is False
    assert "not_started" in (reason or "")


# ----------------------------------------------------------------------
# Disk layout sanity (post-write inspection)
# ----------------------------------------------------------------------


def test_value_lands_at_expected_slot_offset(tmp_path: Path):
    """The byte at offset (slot_idx * slot_bytes) in the right container
    must equal the value we PUT. Verifies the offset math hasn't drifted."""
    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=16 * 1024,
        slot_bytes=1024,  # sub-4K slot
        container_bytes=4 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r.start()
    try:
        r.put(b"k", b"sentinel" + b"\x00" * 50, retention="long", model="m", compat_key="c")
        entry = r.get_entry(b"k", model="m", compat_key="c")
        assert entry is not None

        container_path = tmp_path / "containers" / f"{entry.container_idx:04d}.bin"
        with container_path.open("rb") as f:
            f.seek(entry.slot_idx * 1024)
            on_disk = f.read(entry.size_bytes)
        assert on_disk == b"sentinel" + b"\x00" * 50
    finally:
        r.shutdown()


def test_snapshot_json_is_well_formed(tmp_path: Path):
    """After shutdown, the snapshot file must be parseable JSON with
    the documented schema (version, geometry, checksum, entries)."""
    r = _make_region(tmp_path)
    r.start()
    r.put(b"k", b"v", retention="long", model="m", compat_key="c")
    r.shutdown()

    data = json.loads((tmp_path / "index.snapshot.json").read_bytes())
    assert data["version"] == 1
    assert "geometry" in data
    assert data["geometry"]["slot_bytes"] > 0
    assert data["geometry"]["container_bytes"] > 0
    assert data["checksum"].startswith("sha256:")
    assert len(data["entries"]) == 1
    assert data["entries"][0]["key_hex"] == b"k".hex()
    assert data["entries"][0]["size"] == 1


# ----------------------------------------------------------------------
# Concurrency smoke
# ----------------------------------------------------------------------


def test_concurrent_puts_serialize_via_internal_lock(tmp_path: Path):
    """The region's internal lock serializes index mutations. We don't
    assert ordering, just no corruption + no lost entries."""
    import threading

    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=1024 * 1024,
        slot_bytes=4 * 1024,
        container_bytes=64 * 1024,
        sync_writes=False,
        # This test specifically targets the index-mutation lock, not
        # the IO mode — keep buffered to isolate the variable.
        # O_DIRECT concurrent behaviour is exercised in
        # `test_o_direct_concurrent_reads_write`.
        o_direct=False,
    )
    r.start()
    try:
        n_threads = 8
        n_per_thread = 16

        def worker(tid: int) -> None:
            for i in range(n_per_thread):
                key = f"t{tid}-i{i}".encode()
                value = struct.pack("<I", tid) + struct.pack("<I", i) + b"\x00" * 100
                ok, _ = r.put(key, value, retention="long", model="m", compat_key="c")
                assert ok

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert r.entries_count == n_threads * n_per_thread
        # Spot-check that values round-trip correctly.
        for tid in range(n_threads):
            for i in range(n_per_thread):
                key = f"t{tid}-i{i}".encode()
                val = r.get_bytes(key, model="m", compat_key="c")
                assert val is not None
                got_tid, got_i = struct.unpack("<II", val[:8])
                assert got_tid == tid
                assert got_i == i
    finally:
        r.shutdown()


def test_entry_dataclass_slot_global_helper():
    e = TablespaceEntry(
        key=b"k",
        container_idx=2,
        slot_idx=5,
        size_bytes=100,
        retention="long",
    )
    # If slots_per_container = 16: global = 2*16+5 = 37.
    assert e.slot_global(16) == 37


# ----------------------------------------------------------------------
# O_DIRECT path (B.2)
# ----------------------------------------------------------------------


def _o_direct_supported(tmp_path: Path) -> bool:
    """Probe whether the underlying filesystem allows opening a file
    with O_DIRECT. tmpfs in some setups doesn't; we skip those tests
    rather than fail."""
    import os as _os

    probe = tmp_path / ".odirect_probe"
    try:
        fd = _os.open(probe, _os.O_RDWR | _os.O_CREAT | _os.O_DIRECT, 0o600)
    except OSError:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
        return False
    _os.close(fd)
    probe.unlink(missing_ok=True)
    return True


def test_o_direct_construction_rejects_unaligned_slot_bytes(tmp_path: Path):
    """O_DIRECT requires slot_bytes to be a 4 KB multiple."""
    with pytest.raises(ValueError, match="multiple of 4096"):
        TablespaceLongRegion(
            tmp_path,
            max_bytes=64 * 1024,
            slot_bytes=1024,  # NOT 4 KB-aligned
            container_bytes=16 * 1024,
            sync_writes=False,
            o_direct=True,
        )


def test_o_direct_round_trip(tmp_path: Path):
    """End-to-end PUT/GET with O_DIRECT — same correctness as the
    buffered path."""
    if not _o_direct_supported(tmp_path):
        pytest.skip("filesystem does not support O_DIRECT")
    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=4096,
        container_bytes=16 * 1024,
        sync_writes=False,
        o_direct=True,
    )
    r.start()
    try:
        payload = b"o-direct payload " + b"\xaa" * 64
        ok, reason = r.put(b"k", payload, retention="long", model="m", compat_key="c")
        assert ok, reason
        # Read it back.
        got = r.get_bytes(b"k", model="m", compat_key="c")
        assert got == payload
    finally:
        r.shutdown()


def test_o_direct_survives_restart(tmp_path: Path):
    """O_DIRECT writes must still land on disk and replay correctly —
    the journal + snapshot path doesn't care about the IO mode."""
    if not _o_direct_supported(tmp_path):
        pytest.skip("filesystem does not support O_DIRECT")

    payload = b"persistent o_direct " + b"\xbb" * 200

    r1 = TablespaceLongRegion(
        tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=4096,
        container_bytes=16 * 1024,
        sync_writes=True,  # force fsync on journal so a `del r1` simulates a crash
        o_direct=True,
    )
    r1.start()
    r1.put(b"k", payload, retention="long", model="m", compat_key="c")
    del r1  # ungraceful — no shutdown, no snapshot

    r2 = TablespaceLongRegion(
        tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=4096,
        container_bytes=16 * 1024,
        sync_writes=False,
        o_direct=True,
    )
    r2.start()
    try:
        got = r2.get_bytes(b"k", model="m", compat_key="c")
        assert got == payload
    finally:
        r2.shutdown()


def test_o_direct_alignment_padding_invisible_to_reader(tmp_path: Path):
    """Even though O_DIRECT pads writes to slot_bytes, get_bytes must
    return EXACTLY the bytes that were put — not the zero-padded slot."""
    if not _o_direct_supported(tmp_path):
        pytest.skip("filesystem does not support O_DIRECT")
    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=4096,
        container_bytes=16 * 1024,
        sync_writes=False,
        o_direct=True,
    )
    r.start()
    try:
        # Three different sizes to exercise the padding boundary.
        for size in (1, 100, 4095):
            payload = b"X" * size
            r.put(f"k{size}".encode(), payload, retention="long", model="m", compat_key="c")
            got = r.get_bytes(f"k{size}".encode(), model="m", compat_key="c")
            assert got == payload, (
                f"size={size}: O_DIRECT padding leaked through; "
                f"got len={len(got) if got else None}, want {size}"
            )
    finally:
        r.shutdown()


def test_o_direct_concurrent_reads_write(tmp_path: Path):
    """Concurrent O_DIRECT operations must not corrupt each other.
    Each call allocates its own aligned buffer — verifies that
    per-call allocation is in fact thread-safe."""
    if not _o_direct_supported(tmp_path):
        pytest.skip("filesystem does not support O_DIRECT")
    import threading as _threading

    r = TablespaceLongRegion(
        tmp_path,
        max_bytes=64 * 4096,
        slot_bytes=4096,
        container_bytes=16 * 4096,
        sync_writes=False,
        o_direct=True,
    )
    r.start()
    try:
        n_threads = 4
        n_per_thread = 8

        def worker(tid: int) -> None:
            for i in range(n_per_thread):
                key = f"t{tid}-i{i}".encode()
                value = struct.pack("<II", tid, i) + b"\x00" * 100
                ok, _ = r.put(value=value, key=key, retention="long", model="m", compat_key="c")
                assert ok
                got = r.get_bytes(key, model="m", compat_key="c")
                assert got == value

        threads = [_threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert r.entries_count == n_threads * n_per_thread
    finally:
        r.shutdown()


def test_get_bytes_parallelizes_under_concurrent_callers(tmp_path: Path, monkeypatch):
    """get_bytes must release the dict lock before issuing pread, so
    N concurrent readers measure wall ≈ one read, not N × one read.

    This is the load-bearing precondition for the intra-shard worker
    pool: if pread were inside the lock, sub-workers would serialize
    and the fanout would be a no-op. We verify by monkey-patching
    os.pread to sleep 20 ms; with 8 concurrent callers, wall time must
    be < 4 × 20 ms (= 80 ms) — a 2× safety margin over the ideal 20 ms
    while still flunking any version that holds the lock across the
    syscall (which would be ≥ 8 × 20 = 160 ms).
    """
    import os as _os
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=128 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r.start()
    try:
        # Populate 8 keys, each routed to its own slot.
        n_keys = 8
        payloads = {}
        for i in range(n_keys):
            k = f"k{i}".encode()
            v = f"v{i}".encode() + b"\x00" * 64
            ok, _ = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok
            payloads[k] = v

        # Replace os.pread with a slow shim. The shim returns the real
        # bytes (we still want correctness) but sleeps 20 ms first.
        real_pread = _os.pread
        per_call_delay_s = 0.020  # 20 ms
        call_count = {"n": 0}
        call_lock = threading.Lock()

        def slow_pread(fd, n, offset):
            with call_lock:
                call_count["n"] += 1
            _time.sleep(per_call_delay_s)
            return real_pread(fd, n, offset)

        monkeypatch.setattr(_os, "pread", slow_pread)

        # 8 concurrent get_bytes — if the lock is held over the
        # syscall, wall ≈ 8 × 20 ms = 160 ms. If not, wall ≈ 20 ms +
        # overhead.
        keys = list(payloads.keys())
        with ThreadPoolExecutor(max_workers=n_keys) as ex:
            t0 = _time.perf_counter()
            futures = [ex.submit(r.get_bytes, k, model="m", compat_key="ck") for k in keys]
            results = [f.result() for f in futures]
            elapsed = _time.perf_counter() - t0

        # Correctness: every read got its real payload.
        assert results == [payloads[k] for k in keys]
        assert call_count["n"] == n_keys

        # Parallelism: wall < 4× single-call latency. Serial would be
        # 8× = 160 ms; we allow up to 80 ms for ThreadPoolExecutor
        # scheduling slack. Anything above 80 ms means the dict lock
        # is being held across the syscall — a perf bug we'd want to
        # surface immediately.
        max_acceptable_s = 4 * per_call_delay_s
        assert elapsed < max_acceptable_s, (
            f"concurrent get_bytes wall {elapsed:.3f}s exceeds "
            f"{max_acceptable_s:.3f}s — pread may be running under the dict lock"
        )
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Group commit / batch fsync (NFS-friendly optimization)
#
# When flush_interval_ms > 0, `put` skips inline fsync and a background
# thread fsyncs every interval. Big win for high-fsync-cost backends
# (NFS ~1.2 ms per fsync); negligible for local NVMe.
# ----------------------------------------------------------------------


def test_group_commit_defers_fsync_to_background(tmp_path: Path):
    """In group-commit mode, `put` returns immediately without
    fsyncing. We don't have a direct "did fsync fire?" hook so we
    verify indirectly: dirty_containers set grows on PUT, gets
    cleared after the flusher tick."""
    import time

    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=128 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=50,
    )
    r.start()
    try:
        # PUT marks the key's hash-routed container dirty. We don't
        # assume container 0 here — the hash-distributed allocator
        # places ``b"k"`` into whichever container its blake2b digest
        # picks. Look up the actual container from the entry instead.
        r.put(b"k", b"x" * 100, retention="long", model="m", compat_key="c")
        entry = r.get_entry(b"k", model="m", compat_key="c")
        assert entry is not None
        with r._flush_lock:
            assert entry.container_idx in r._dirty_containers, (
                "PUT should have marked the key's container dirty"
            )

        # After one flush interval the background thread fsyncs and clears.
        time.sleep(0.15)  # 3 × interval window for reliability
        with r._flush_lock:
            assert r._dirty_containers == set(), "flusher should have cleared dirty set"
    finally:
        r.shutdown()


def test_group_commit_off_means_inline_fsync(tmp_path: Path):
    """flush_interval_ms=0 (default) means dirty_containers stays empty
    because PUT fsyncs inline."""
    r = _make_region(tmp_path)  # default flush_interval_ms=0
    r.start()
    try:
        r.put(b"k", b"v", retention="long", model="m", compat_key="c")
        with r._flush_lock:
            assert r._dirty_containers == set(), (
                "with inline fsync (flush_interval_ms=0), dirty_containers should not accumulate"
            )
        # And the flusher thread shouldn't exist.
        assert r._flush_thread is None
    finally:
        r.shutdown()


def test_group_commit_shutdown_flushes_pending(tmp_path: Path):
    """shutdown() must fsync all pending dirty containers before
    closing fds — otherwise group-commit risks losing the last batch
    even on graceful shutdown."""
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=128 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=10000,  # interval long enough that bg flusher won't fire
    )
    r.start()
    r.put(b"k", b"survives-shutdown", retention="long", model="m", compat_key="c")
    # Don't sleep — shut down before the flusher ticks.
    r.shutdown()

    # Open a fresh region against the same dir; the bytes must be readable.
    r2 = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=128 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r2.start()
    try:
        assert r2.get_bytes(b"k", model="m", compat_key="c") == b"survives-shutdown", (
            "shutdown must flush pending writes before closing fds"
        )
    finally:
        r2.shutdown()


def test_group_commit_flush_method_is_idempotent(tmp_path: Path):
    """Calling flush() with nothing dirty is a no-op. Calling it twice
    in a row is fine."""
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=128 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=100,
    )
    r.start()
    try:
        r.flush()  # nothing dirty yet
        r.put(b"k", b"v", retention="long", model="m", compat_key="c")
        r.flush()
        r.flush()  # second call also fine
    finally:
        r.shutdown()


def test_put_refused_while_flusher_failing(tmp_path: Path):
    """When the background flusher hits a
    persistent fsync error (full disk, device disconnected) PUT
    must refuse new writes instead of pretending success while
    data goes nowhere. Use the public flag directly to simulate
    the failure mode."""
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=0,
    )
    r.start()
    try:
        # Initially PUT works.
        ok, _ = r.put(b"k1", b"v1", retention="long", model="m", compat_key="c")
        assert ok

        # Simulate the flusher reporting an error.
        r._flush_error = OSError("ENOSPC: simulated full disk")
        ok, reason = r.put(b"k2", b"v2", retention="long", model="m", compat_key="c")
        assert ok is False
        assert "flusher_failing" in (reason or "")
        assert "ENOSPC" in (reason or "")

        # Clear the error → PUT works again.
        r._flush_error = None
        ok, _ = r.put(b"k3", b"v3", retention="long", model="m", compat_key="c")
        assert ok
    finally:
        r.shutdown()


def test_group_commit_many_writes_amortize_fsyncs(tmp_path: Path):
    """Sanity / no-corruption: do many writes in group-commit mode,
    verify each round-trips correctly. (Throughput improvement is
    backend-dependent; this is the correctness check.)"""
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=256 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=64 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=20,
    )
    r.start()
    try:
        for i in range(20):
            payload = f"payload-{i}".encode() + b"\x00" * 100
            ok, _ = r.put(f"k{i}".encode(), payload, retention="long", model="m", compat_key="c")
            assert ok
        # No need to wait for flush — get reads from RAM pwrite-buffer
        # (which is the kernel page cache); fsync only matters for
        # crash safety, not for in-process reads.
        for i in range(20):
            payload = f"payload-{i}".encode() + b"\x00" * 100
            assert r.get_bytes(f"k{i}".encode(), model="m", compat_key="c") == payload
        assert r.entries_count == 20
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# detect_fs_defaults — per-filesystem auto-tuning
# ----------------------------------------------------------------------


def test_detect_fs_defaults_table_has_wekafs_buffered():
    """Weka must come pre-configured: O_DIRECT off (its writecache
    coalesces RDMA writes; bypassing it costs 5–11× throughput), with
    group commit on (20 ms window)."""
    weka = _FS_DEFAULTS["wekafs"]
    assert weka["o_direct"] is False
    assert weka["flush_interval_ms"] == 20


def test_detect_fs_defaults_table_has_local_fs_o_direct():
    """ext4/xfs/btrfs should default to O_DIRECT on, inline fsync;
    O_DIRECT measured −99% page-cache RAM, fsync is cheap."""
    for fs in ("ext4", "xfs", "btrfs"):
        d = _FS_DEFAULTS[fs]
        assert d["o_direct"] is True, f"{fs} should default to O_DIRECT"
        assert d["flush_interval_ms"] == 0, f"{fs} should default to inline fsync"


def test_detect_fs_defaults_table_has_nfs_grouped():
    """NFS variants get group commit (38× write speedup)."""
    for fs in ("nfs", "nfs4"):
        d = _FS_DEFAULTS[fs]
        assert d["o_direct"] is True
        assert d["flush_interval_ms"] == 20


def test_detect_fs_defaults_returns_actual_path_fstype(tmp_path: Path):
    """End-to-end: calling on a real path returns a dict with the
    three expected keys + a plausible fstype string."""
    d = detect_fs_defaults(tmp_path)
    assert "fstype" in d and isinstance(d["fstype"], str) and len(d["fstype"]) > 0
    assert isinstance(d["o_direct"], bool)
    assert isinstance(d["flush_interval_ms"], int)
    assert d["flush_interval_ms"] >= 0


def test_detect_fs_defaults_unknown_falls_back_safe(tmp_path: Path, monkeypatch):
    """Unknown filesystem → safe conservative default (buffered, inline)."""
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "made-up-fs")
    d = detect_fs_defaults(tmp_path)
    assert d["fstype"] == "made-up-fs"
    assert d["o_direct"] is False
    assert d["flush_interval_ms"] == 0


# ----------------------------------------------------------------------
# TablespaceLongRegion auto-detect integration
# ----------------------------------------------------------------------


def test_auto_detect_resolves_o_direct_none_at_start(tmp_path: Path, monkeypatch):
    """When constructor receives o_direct=None, start() resolves it
    via detect_fs_defaults. Verify against a stubbed fstype."""
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "wekafs")
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=None,
        flush_interval_ms=0,
    )
    assert r._o_direct is None  # not resolved yet at __init__
    r.start()
    try:
        assert r._o_direct is False  # wekafs → buffered
        assert r._flush_interval_ms == 0  # explicit; not auto-detected
    finally:
        r.shutdown()


def test_auto_detect_resolves_flush_interval_none_at_start(tmp_path: Path, monkeypatch):
    """flush_interval_ms=None auto-detects independently of o_direct."""
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "nfs")
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=None,
    )
    r.start()
    try:
        assert r._o_direct is False  # explicit; not auto-detected
        assert r._flush_interval_ms == 20  # nfs → grouped
    finally:
        r.shutdown()


def test_auto_detect_both_none_resolves_both(tmp_path: Path, monkeypatch):
    """When both fields are None, both get resolved from the same
    detected fstype."""
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "wekafs")
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=None,
        flush_interval_ms=None,
    )
    r.start()
    try:
        assert r._o_direct is False
        assert r._flush_interval_ms == 20
    finally:
        r.shutdown()


def test_explicit_values_never_auto_detect(tmp_path: Path, monkeypatch):
    """Operator-supplied o_direct=False / flush_interval_ms=0 must
    win even when the detected fstype recommends different values.
    Guards against the 'I disabled it explicitly and it came back on'
    failure mode."""
    # Stub detection to wekafs (which recommends o_direct=False,
    # flush_interval_ms=20). But constructor explicitly says
    # o_direct=False AND flush_interval_ms=0 — auto-detect MUST NOT
    # override either.
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "wekafs")
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=8 * 1024,
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=False,
        flush_interval_ms=0,
    )
    r.start()
    try:
        assert r._o_direct is False
        assert r._flush_interval_ms == 0  # NOT 20
    finally:
        r.shutdown()


def test_auto_detect_o_direct_true_validates_alignment(tmp_path: Path, monkeypatch):
    """If auto-detect picks o_direct=True but slot_bytes isn't 4 KB-
    aligned, start() must raise — same invariant as the explicit case,
    just enforced later."""
    monkeypatch.setattr("infera.kvd.tablespace._detect_fstype", lambda p: "ext4")
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=64 * 1024,
        slot_bytes=4097,  # not a 4 KB multiple
        container_bytes=32 * 1024,
        sync_writes=False,
        o_direct=None,
        flush_interval_ms=0,
    )
    with pytest.raises(ValueError, match="o_direct=True"):
        r.start()


# ----------------------------------------------------------------------
# Hash-distributed container allocation
# ----------------------------------------------------------------------


def _make_hash_region(tmp_path: Path, **kw) -> TablespaceLongRegion:
    """Region geometry tuned for the hash-distribution tests: 8 containers
    × 16 slots × 4 KB = 128 slots, 512 KB total. The 8-container shape
    is the load-bearing property — the same layout the striped
    bench uses (8 NVMe shards × N containers per shard).
    """
    return TablespaceLongRegion(
        path=tmp_path,
        max_bytes=kw.pop("max_bytes", 8 * 16 * 4096),
        slot_bytes=kw.pop("slot_bytes", 4096),
        container_bytes=kw.pop("container_bytes", 16 * 4096),
        sync_writes=kw.pop("sync_writes", False),
        o_direct=kw.pop("o_direct", False),
        flush_interval_ms=kw.pop("flush_interval_ms", 0),
    )


def test_alloc_distributes_across_containers_by_key_hash(tmp_path: Path):
    """PUT 128 distinct keys into an 8-container region; each container
    should end up with roughly 128/8 = 16 slots filled. The hash is
    blake2b which has good distribution at any sample size we care
    about — assert per-container occupancy is within ±30% of the
    expected mean (a generous slack that still flunks the legacy
    sequential-fill allocator, which would put 16/16/16/16/16/16/16/16
    only if 128 PUTs perfectly fill the region, but would put all 128
    into container 0..7 in strict order — first 16 to c0, next 16 to
    c1, etc.).
    """
    r = _make_hash_region(tmp_path)
    r.start()
    try:
        # 128 PUTs into 128-slot region; fills exactly.
        n_keys = 128
        for i in range(n_keys):
            ok, reason = r.put(
                f"key-{i:04d}".encode(),
                b"V" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok, reason

        # Inspect per-container occupancy.
        per_container_counts = [
            r._allocator.per_container_allocated(ci) for ci in range(r.num_containers)
        ]
        total = sum(per_container_counts)
        assert total == n_keys
        expected = n_keys / r.num_containers  # 16
        max_dev = expected * 0.30  # within 30% of mean

        # Stronger structural check: NO container should be empty and
        # no container should hold more than ~2x the mean. The legacy
        # sequential allocator would have container 0 with 16 and the
        # rest with 16, 16, 16... only because the region fills exactly.
        # If we shrank the workload to 24 keys, sequential would put 16
        # in c0 and 8 in c1 with c2..c7 EMPTY — we check that variant
        # below.
        assert all(c > 0 for c in per_container_counts), (
            f"hash distribution failed — some containers empty: {per_container_counts}"
        )
        for ci, c in enumerate(per_container_counts):
            assert abs(c - expected) <= max_dev, (
                f"container {ci}: {c} slots, expected {expected} ±{max_dev:.1f} "
                f"(all={per_container_counts})"
            )
    finally:
        r.shutdown()


def test_alloc_legacy_would_cluster_into_container_0(tmp_path: Path):
    """The exact failure mode we're fixing: with only 24 PUTs (= 1.5
    containers' worth) into an 8-container region, the legacy
    sequential-fill allocator would put the first 16 in c0 and the
    last 8 in c1, leaving c2..c7 EMPTY. The hash-distributed allocator
    spreads them across all 8 containers — every container should
    have at least one entry by the time we hit 24 keys.
    """
    r = _make_hash_region(tmp_path)
    r.start()
    try:
        for i in range(24):
            ok, _ = r.put(
                f"key-{i:04d}".encode(),
                b"V" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok

        per_container_counts = [
            r._allocator.per_container_allocated(ci) for ci in range(r.num_containers)
        ]
        n_nonempty = sum(1 for c in per_container_counts if c > 0)
        # 24 keys / 8 containers with a well-distributed hash → every
        # container hit. Legacy sequential fill would give n_nonempty=2.
        assert n_nonempty >= 6, (
            f"hash should hit most containers; got {n_nonempty}/8 nonempty: {per_container_counts}"
        )
    finally:
        r.shutdown()


def test_alloc_fallback_when_target_container_full(tmp_path: Path, caplog):
    """Fill container 0 directly via the allocator's mark_used; then
    insert a key whose hash routes to container 0. The slot must land
    in a DIFFERENT container (fallback probe), and the one-shot WARN
    must fire."""
    import logging

    r = _make_hash_region(tmp_path)
    r.start()
    try:
        spc = r._slots_per_container  # 16
        # Mark every slot in container 0 as used directly. (Skip the
        # full PUT path so the test isolates the alloc fallback, not
        # the IO write loop.)
        for local in range(spc):
            assert r._allocator.mark_used(local) is True
        assert r._allocator.per_container_allocated(0) == spc

        # Find a key that hashes to container 0. blake2b is
        # deterministic so we just probe until we hit one — should be
        # within a few tries since 1/8 of all keys go to container 0.
        target_key: bytes | None = None
        for i in range(1024):
            k = f"probe-{i:04d}".encode()
            composite = b"m" + b"\x00" + b"ck" + b"\x00" + k
            h = int.from_bytes(
                __import__("hashlib").blake2b(composite, digest_size=8).digest(), "big"
            )
            if h % r.num_containers == 0:
                target_key = k
                break
        assert target_key is not None, "expected at least one probe key to hash to c0"

        with caplog.at_level(logging.WARNING, logger="infera.kvd.tablespace"):
            ok, reason = r.put(target_key, b"X" * 64, retention="long", model="m", compat_key="ck")
        assert ok, reason

        entry = r.get_entry(target_key, model="m", compat_key="ck")
        assert entry is not None
        assert entry.container_idx != 0, (
            "fallback should have moved off the full target container 0"
        )

        # Exactly one WARN about the fallback firing.
        fallback_warnings = [
            rec for rec in caplog.records if "hash-target container" in rec.message
        ]
        assert len(fallback_warnings) == 1, (
            f"expected one-shot fallback WARN, got {len(fallback_warnings)}"
        )

        # PUT a second key that also hashes to c0 — same fallback path,
        # but the WARN must NOT re-fire (sticky one-shot).
        caplog.clear()
        second_key: bytes | None = None
        for i in range(1024, 4096):
            k = f"probe-{i:04d}".encode()
            composite = b"m" + b"\x00" + b"ck" + b"\x00" + k
            h = int.from_bytes(
                __import__("hashlib").blake2b(composite, digest_size=8).digest(), "big"
            )
            if h % r.num_containers == 0:
                second_key = k
                break
        assert second_key is not None
        with caplog.at_level(logging.WARNING, logger="infera.kvd.tablespace"):
            ok, _ = r.put(second_key, b"X" * 64, retention="long", model="m", compat_key="ck")
        assert ok
        fallback_warnings_2 = [
            rec for rec in caplog.records if "hash-target container" in rec.message
        ]
        assert fallback_warnings_2 == [], (
            "fallback WARN should be one-shot; subsequent fallbacks must stay quiet"
        )
    finally:
        r.shutdown()


def test_restart_preserves_per_container_state(tmp_path: Path):
    """Per-container allocated counts must survive a restart, AND the
    hash routing must keep working after restart — i.e. new PUTs
    continue to land in their hash-target containers (not all of them
    in container 0)."""
    n_initial = 64

    r1 = _make_hash_region(tmp_path)
    r1.start()
    pre_counts: list[int] = []
    pre_entries: list[tuple[bytes, int]] = []  # (key, container_idx)
    try:
        for i in range(n_initial):
            ok, _ = r1.put(
                f"persist-{i:04d}".encode(),
                b"P" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok
        pre_counts = [r1._allocator.per_container_allocated(ci) for ci in range(r1.num_containers)]
        for i in range(n_initial):
            k = f"persist-{i:04d}".encode()
            entry = r1.get_entry(k, model="m", compat_key="ck")
            assert entry is not None
            pre_entries.append((k, entry.container_idx))
    finally:
        r1.shutdown()

    # Restart.
    r2 = _make_hash_region(tmp_path)
    r2.start()
    try:
        post_counts = [r2._allocator.per_container_allocated(ci) for ci in range(r2.num_containers)]
        assert post_counts == pre_counts, (
            f"per-container counts must survive restart: pre={pre_counts} post={post_counts}"
        )

        # Existing entries point at the same containers as before.
        for k, expected_ci in pre_entries:
            entry = r2.get_entry(k, model="m", compat_key="ck")
            assert entry is not None
            assert entry.container_idx == expected_ci, (
                f"restart moved key {k!r} from c{expected_ci} to c{entry.container_idx}"
            )

        # New PUTs continue to hash-distribute (assert: no single container
        # absorbs all of them).
        n_new = 32
        for i in range(n_new):
            ok, _ = r2.put(
                f"new-{i:04d}".encode(),
                b"N" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok
        delta = [
            r2._allocator.per_container_allocated(ci) - pre_counts[ci]
            for ci in range(r2.num_containers)
        ]
        assert sum(delta) == n_new
        n_nonempty_new = sum(1 for d in delta if d > 0)
        assert n_nonempty_new >= 6, (
            f"post-restart hash routing should still spread across containers; "
            f"only {n_nonempty_new}/8 got new entries: delta={delta}"
        )
    finally:
        r2.shutdown()


def test_legacy_data_loads_with_sequential_layout(tmp_path: Path):
    """A tablespace dir whose snapshot was written by an OLDER allocator
    (sequential fill — first 16 in c0, next 16 in c1, ...) must still
    load and serve reads correctly under the new allocator. We
    construct the snapshot by hand to ensure we exercise the exact
    legacy layout, not a hash-distributed coincidence.
    """
    # Geometry: 8 containers × 16 slots × 4 KB.
    slot_bytes = 4096
    container_bytes = 16 * slot_bytes
    max_bytes = 8 * container_bytes
    n_containers = 8
    slots_per_container = 16

    # Stage container files with deterministic payloads at the legacy
    # offsets — first 16 keys in c0, next 16 in c1, ...
    containers_dir = tmp_path / "containers"
    containers_dir.mkdir(parents=True)
    payloads: dict[bytes, tuple[int, int, bytes]] = {}
    # Only fill c0 and c1 with the sequential pattern; that's enough
    # to prove the legacy layout loads (the failure mode would be
    # decoding c0 entries as if they were hash-distributed elsewhere).
    n_keys = 24
    for i in range(n_keys):
        container_idx = i // slots_per_container
        slot_idx = i % slots_per_container
        key = f"legacy-{i:04d}".encode()
        payload = (
            b"LEGACY"
            + i.to_bytes(2, "big")
            + b"\x00" * (slot_bytes - 8)  # pad to full slot (writes are slot-aligned in
            # buffered mode; we only read the recorded size_bytes below)
        )[:slot_bytes]
        cp = containers_dir / f"{container_idx:04d}.bin"
        with cp.open("ab") as f:
            # Pad up to the slot offset, then write the payload.
            current = f.tell()
            offset = slot_idx * slot_bytes
            if current < offset:
                f.write(b"\x00" * (offset - current))
            f.write(payload)
        payloads[key] = (container_idx, slot_idx, payload)
    # Ensure each container file is full-size (preallocation-equivalent).
    for ci in range(n_containers):
        cp = containers_dir / f"{ci:04d}.bin"
        if not cp.exists():
            cp.write_bytes(b"\x00" * container_bytes)
        else:
            current = cp.stat().st_size
            if current < container_bytes:
                with cp.open("ab") as f:
                    f.write(b"\x00" * (container_bytes - current))

    # Build a snapshot file with the sequential entries.
    snapshot_entries = []
    for key, (ci, si, payload) in payloads.items():
        # We declare size as the actual payload bytes (matches what a
        # sequential PUT would have recorded). The read path slices to
        # this size.
        snapshot_entries.append(
            {
                "key_hex": key.hex(),
                "container": ci,
                "slot": si,
                "size": len(payload),
                "retention": "long",
                "model": "m",
                "compat_key": "ck",
                "metadata": {},
            }
        )
    entries_json = json.dumps(snapshot_entries, sort_keys=True, separators=(",", ":")).encode()
    import hashlib as _hashlib

    snapshot_payload = {
        "version": 1,
        "geometry": {"slot_bytes": slot_bytes, "container_bytes": container_bytes},
        "checksum": f"sha256:{_hashlib.sha256(entries_json).hexdigest()}",
        "entries": snapshot_entries,
    }
    (tmp_path / "index.snapshot.json").write_bytes(
        json.dumps(snapshot_payload, sort_keys=True, separators=(",", ":")).encode()
    )

    # Now open a region against the dir. The new (hash-distributed)
    # allocator should mark_used the legacy slots and serve reads.
    r = TablespaceLongRegion(
        path=tmp_path,
        max_bytes=max_bytes,
        slot_bytes=slot_bytes,
        container_bytes=container_bytes,
        sync_writes=False,
        o_direct=False,
    )
    r.start()
    try:
        assert r.entries_count == n_keys
        # The first 16 entries are in c0; the next 8 are in c1. The
        # allocator's per-container counts reflect that LEGACY layout.
        assert r._allocator.per_container_allocated(0) == slots_per_container
        assert r._allocator.per_container_allocated(1) == n_keys - slots_per_container
        for ci in range(2, n_containers):
            assert r._allocator.per_container_allocated(ci) == 0

        # Reads return the correct bytes.
        for key, (ci, si, expected_payload) in payloads.items():
            got = r.get_bytes(key, model="m", compat_key="ck")
            assert got == expected_payload, (
                f"key {key!r} (legacy c{ci}/s{si}) read mismatch: got={got[:16] if got else None}"
            )
    finally:
        r.shutdown()


def test_get_bytes_with_hash_distribution_parallelizes(tmp_path: Path, monkeypatch):
    """Extension of test_get_bytes_parallelizes_under_concurrent_callers:
    confirm that with hash-distributed placement, concurrent
    get_bytes calls hit different container fds (not just different
    slot offsets in the same fd). The structural fanout — across
    inodes — is what unlocks the per-device parallelism on real NVMe.
    """
    import os as _os
    import threading
    import time as _time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    r = _make_hash_region(tmp_path)
    r.start()
    try:
        # PUT 8 keys; with 8 containers and a well-distributed hash,
        # we expect them to land in (mostly) different containers.
        n_keys = 32
        keys = []
        for i in range(n_keys):
            k = f"par-{i:04d}".encode()
            v = (b"PAR" + i.to_bytes(2, "big")) + b"\x00" * 64
            ok, _ = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok
            keys.append(k)

        # Per-container distribution of these keys.
        fd_for_key = {}
        per_container_keys = Counter()
        for k in keys:
            entry = r.get_entry(k, model="m", compat_key="ck")
            assert entry is not None
            fd_for_key[k] = entry.container_idx
            per_container_keys[entry.container_idx] += 1
        # Hash distribution should hit at least 6/8 containers with
        # 32 keys (well below a "single-container clustering" failure).
        assert len(per_container_keys) >= 6, (
            f"hash distribution should fan out 32 keys across ≥6 containers; "
            f"actual: {dict(per_container_keys)}"
        )

        # Slow pread shim to expose serialization: each call sleeps
        # 20 ms. If all reads serialized on a single fd's inode lock,
        # 32 reads would take ≥640 ms. With fanout across 6+ container
        # fds we expect ≤ 4× single-call (= 80 ms).
        real_pread = _os.pread
        per_call_delay_s = 0.020
        fd_call_count = Counter()
        call_lock = threading.Lock()

        def slow_pread(fd, n, offset):
            with call_lock:
                fd_call_count[fd] += 1
            _time.sleep(per_call_delay_s)
            return real_pread(fd, n, offset)

        monkeypatch.setattr(_os, "pread", slow_pread)

        with ThreadPoolExecutor(max_workers=n_keys) as ex:
            t0 = _time.perf_counter()
            futures = [ex.submit(r.get_bytes, k, model="m", compat_key="ck") for k in keys]
            results = [f.result() for f in futures]
            elapsed = _time.perf_counter() - t0

        # Correctness: all reads succeeded.
        assert all(v is not None for v in results)

        # Different fds were touched (the load-bearing structural assertion).
        assert len(fd_call_count) >= 6, (
            f"expected ≥6 distinct fds touched (one per container hit); "
            f"got {len(fd_call_count)}: {dict(fd_call_count)}"
        )

        # Wall time stays well under the serial baseline. We allow
        # generous slack because chunking 32 reads across 8 inodes still
        # means ~4 reads queued per inode (~80 ms each), plus thread
        # pool scheduling overhead. 12× single-call is the cutoff —
        # serial would be 32 × 20 = 640 ms (= 32×), so 12× still
        # demonstrates the fanout.
        max_acceptable_s = 12 * per_call_delay_s
        assert elapsed < max_acceptable_s, (
            f"concurrent get_bytes wall {elapsed:.3f}s exceeds {max_acceptable_s:.3f}s — "
            f"hash-fanout reads aren't parallelizing across container fds"
        )
    finally:
        r.shutdown()


def test_alloc_with_no_key_hash_falls_back_to_sequential(tmp_path: Path):
    """The ``alloc(None)`` safety hatch must keep the legacy sequential
    behavior for paths that don't have a key — container 0 first, etc.
    """
    r = _make_hash_region(tmp_path)
    r.start()
    try:
        # Direct allocator probe.
        slots = []
        for _ in range(20):
            s = r._allocator.alloc(None)
            assert s is not None
            slots.append(s)
        # First 16 in container 0 (slots 0..15), next 4 in container 1
        # (slots 16..19) — purely sequential.
        assert slots[:16] == list(range(16))
        assert slots[16:20] == list(range(16, 20))
    finally:
        r.shutdown()


def test_allocator_construction_rejects_invalid_geometry():
    """Pathological allocator geometry must raise rather than silently
    misbehave."""
    with pytest.raises(ValueError):
        BitsetAllocator(0)
    with pytest.raises(ValueError):
        BitsetAllocator(64, num_containers=0)
    with pytest.raises(ValueError):
        # 8 containers × 4 slots = 32 < total 64 → undercovers slots.
        BitsetAllocator(64, num_containers=8, slots_per_container=4)


def test_allocator_per_container_allocated_out_of_range_raises():
    a = BitsetAllocator(64, num_containers=8, slots_per_container=8)
    with pytest.raises(ValueError):
        a.per_container_allocated(8)
    with pytest.raises(ValueError):
        a.per_container_allocated(-1)


def test_recompute_state_restores_counts_from_bitmap():
    """After arbitrary mark_used calls, recompute_state derives the
    per-container counts from the bitmap (counts are kept in sync by
    mark_used itself, but recompute is the authoritative path used at
    restart finalization — verify it's idempotent and correct)."""
    a = BitsetAllocator(64, num_containers=8, slots_per_container=8)
    # Mark slots 0, 1 (c0), 17 (c2), 56 (c7).
    a.mark_used(0)
    a.mark_used(1)
    a.mark_used(17)
    a.mark_used(56)
    pre = [a.per_container_allocated(ci) for ci in range(8)]
    a.recompute_state()
    post = [a.per_container_allocated(ci) for ci in range(8)]
    assert pre == post
    assert post == [2, 0, 1, 0, 0, 0, 0, 1]
    assert a.allocated == 4

    # Next alloc with hash-target=2 should reuse a free slot inside c2
    # (slots 16..23). Slot 17 is taken; the cursor should point at the
    # byte containing slot 16 (the first free) so we get slot 16.
    s = a.alloc(2)  # hash % 8 == 2
    assert s == 16
