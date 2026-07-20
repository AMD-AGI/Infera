###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Integration: HostStore + TablespaceLongRegion.

Validates that the tablespace region drops into HostStore as a
replacement for LongStorageRegion. Mirrors the long-region tests in
test_store_ssd.py — same scenarios, different long-region implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infera.kvd.ssd import SpilloverRegion
from infera.kvd.store import HostStore
from infera.kvd.tablespace import TablespaceLongRegion


def _key(s: str) -> bytes:
    return s.encode("ascii").ljust(8, b"\x00")


@pytest.fixture
def store_with_tablespace_long(tmp_path: Path):
    """HostStore + SpilloverRegion + TablespaceLongRegion. Tiny RAM
    (20 bytes), tiny slot (16 bytes) — same RAM budget as the
    file-per-block integration test so the eviction scenarios are
    directly comparable."""
    spillover = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover.start()
    long_region = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=128,
        slot_bytes=16,
        container_bytes=64,
        sync_writes=False,
        o_direct=False,
    )
    long_region.start()
    store = HostStore(max_bytes=20, spillover=spillover, long_region=long_region)
    return store, tmp_path, long_region


# ----------------------------------------------------------------------
# Long SET write-through to tablespace region
# ----------------------------------------------------------------------


def test_long_set_writes_to_tablespace_region(store_with_tablespace_long):
    """A retention=long SET should land in BOTH host RAM and the
    tablespace long region — same contract as LongStorageRegion."""
    store, _, long_region = store_with_tablespace_long
    accepted, _ = store.set(_key("a"), b"x" * 10, retention="long")
    assert accepted

    # RAM has it.
    assert store.get(_key("a")).value == b"x" * 10
    # Tablespace has it.
    assert long_region.used_bytes == 10
    assert long_region.entries_count == 1


def test_long_block_recoverable_after_ram_eviction(store_with_tablespace_long):
    """RAM evicts → next GET pulls from the tablespace region. Headline
    behavior that makes the SSD tier worth having."""
    store, _, _ = store_with_tablespace_long

    # 20-byte RAM; two 10-byte blocks fit.
    store.set(_key("a"), b"x" * 10, retention="long")
    store.set(_key("b"), b"y" * 10, retention="long")

    # Third long block evicts the LRU from RAM. Long-region keeps both.
    store.set(_key("c"), b"z" * 10, retention="long")

    # GET 'a' — not in RAM anymore, must be served by long region.
    entry = store.get(_key("a"))
    assert entry is not None
    assert entry.value == b"x" * 10


def test_short_block_evicted_to_spillover_when_long_region_full(store_with_tablespace_long):
    """Tablespace long region only takes 'long' retention; short blocks
    bounce to spillover on RAM eviction — same as the file-per-block
    integration."""
    store, _, _ = store_with_tablespace_long
    # Fill RAM with short blocks. capacity = 20 bytes; 10-byte each → 2 fit.
    store.set(_key("a"), b"x" * 10, retention="short")
    store.set(_key("b"), b"y" * 10, retention="short")
    assert store.spillover_bytes == 0  # nothing evicted yet

    # Insert c → evict 'a' to spillover.
    store.set(_key("c"), b"z" * 10, retention="short")
    assert store.spillover_bytes == 10

    # GET 'a' still works (served from spillover).
    assert store.get(_key("a")).value == b"x" * 10


def test_exists_through_host_store_finds_tablespace_entries(store_with_tablespace_long):
    """`HostStore.exists` consults SSD tiers — a long block that's been
    evicted from RAM must still report present."""
    store, _, _ = store_with_tablespace_long

    store.set(_key("a"), b"x" * 10, retention="long")
    store.set(_key("b"), b"y" * 10, retention="long")
    store.set(_key("c"), b"z" * 10, retention="long")  # evicts oldest from RAM

    present = store.exists([_key("a"), _key("b"), _key("c"), _key("missing")])
    assert present == [True, True, True, False]


# ----------------------------------------------------------------------
# Restart survival via tablespace
# ----------------------------------------------------------------------


def test_long_blocks_survive_restart_via_tablespace(tmp_path: Path):
    """The headline persistence property — but via the tablespace
    region instead of the file-per-block manifest."""
    # Phase A: write a long block.
    spillover_a = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_a.start()
    long_a = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=256,
        slot_bytes=64,
        container_bytes=128,
        sync_writes=False,
        o_direct=False,
    )
    long_a.start()
    store_a = HostStore(max_bytes=20, spillover=spillover_a, long_region=long_a)
    store_a.set(_key("durable"), b"important-data", retention="long", model="m1", compat_key="ck1")
    assert store_a.get(_key("durable"), model="m1", compat_key="ck1").value == b"important-data"
    # Graceful shutdown writes snapshot.
    long_a.shutdown()

    # Phase B: rebuild from the same SSD dirs.
    spillover_b = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_b.start()
    long_b = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=256,
        slot_bytes=64,
        container_bytes=128,
        sync_writes=False,
        o_direct=False,
    )
    long_b.start()
    try:
        store_b = HostStore(max_bytes=20, spillover=spillover_b, long_region=long_b)

        entry = store_b.get(_key("durable"), model="m1", compat_key="ck1")
        assert entry is not None
        assert entry.value == b"important-data"
    finally:
        long_b.shutdown()


def test_long_blocks_survive_ungraceful_restart_via_journal(tmp_path: Path):
    """Even WITHOUT a graceful shutdown (no snapshot written), the
    journal alone reconstructs the index — matches the crash-safety
    promise."""
    spillover_a = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_a.start()
    long_a = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=256,
        slot_bytes=64,
        container_bytes=128,
        sync_writes=True,
        o_direct=False,
    )
    long_a.start()
    store_a = HostStore(max_bytes=20, spillover=spillover_a, long_region=long_a)
    store_a.set(_key("crash"), b"survives-crash", retention="long", model="m1", compat_key="ck1")
    # Do NOT call shutdown. Drop the references — simulates a kill.
    del store_a
    del long_a

    # Rebuild — only the journal is on disk, no snapshot.
    long_b = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=256,
        slot_bytes=64,
        container_bytes=128,
        sync_writes=False,
        o_direct=False,
    )
    long_b.start()
    spillover_b = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_b.start()
    try:
        store_b = HostStore(max_bytes=20, spillover=spillover_b, long_region=long_b)
        entry = store_b.get(_key("crash"), model="m1", compat_key="ck1")
        assert entry is not None
        assert entry.value == b"survives-crash"
    finally:
        long_b.shutdown()


# ----------------------------------------------------------------------
# clear() through HostStore
# ----------------------------------------------------------------------


def test_clear_drops_tablespace_entries(store_with_tablespace_long):
    store, _, long_region = store_with_tablespace_long
    store.set(_key("a"), b"x" * 10, retention="long")
    store.set(_key("b"), b"y" * 10, retention="short")
    store.set(_key("c"), b"z" * 10, retention="short")  # evicts b → spillover

    assert long_region.used_bytes > 0
    assert store.spillover_bytes > 0

    store.clear()

    assert long_region.used_bytes == 0
    assert store.spillover_bytes == 0
    assert store.get(_key("a")) is None
    assert store.get(_key("b")) is None


# ----------------------------------------------------------------------
# A larger scenario — many distinct keys, both small and slot-saturating
# ----------------------------------------------------------------------


def test_many_distinct_keys_through_tablespace(tmp_path: Path):
    """Push 32 distinct keys into a long region that has 16 slots —
    half must be evicted. Eviction picks LRU; remaining 16 keys all
    readable through HostStore."""
    spillover = SpilloverRegion(tmp_path / "spill", max_bytes=10_000)
    spillover.start()
    long_region = TablespaceLongRegion(
        tmp_path / "long",
        max_bytes=16 * 64,  # 16 slots × 64 bytes
        slot_bytes=64,
        container_bytes=4 * 64,
        sync_writes=False,
        o_direct=False,
    )
    long_region.start()
    store = HostStore(max_bytes=2 * 64, spillover=spillover, long_region=long_region)

    try:
        for i in range(32):
            ok, _ = store.set(f"k{i:02d}".encode(), b"v" * 50, retention="long")
            assert ok
        assert long_region.entries_count == 16  # capped at slot count

        # 16 most-recently-used keys remain. Roughly k16..k31.
        kept = []
        evicted = []
        for i in range(32):
            ent = store.get(f"k{i:02d}".encode())
            if ent is not None:
                kept.append(i)
            else:
                evicted.append(i)
        assert len(kept) == 16
        # First-inserted keys are the LRU victims.
        assert max(evicted) < min(kept)
    finally:
        long_region.shutdown()
