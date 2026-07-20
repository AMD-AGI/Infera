###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Integration tests: HostStore wired to SpilloverRegion + LongStorageRegion.

These tests verify the 3-tier coordination:

    SET retention=long  → host RAM + long region (write_through)
    SET retention=short → host RAM only
    RAM eviction of short → write to spillover region
    RAM eviction of long  → drop (already on long SSD)
    GET → host RAM → long region → spillover region → miss
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infera.kvd.ssd import LongStorageRegion, SpilloverRegion
from infera.kvd.store import HostStore


def _key(s: str) -> bytes:
    return s.encode("ascii").ljust(8, b"\x00")


@pytest.fixture
def store_with_ssd(tmp_path: Path):
    """HostStore wired with both SSD regions. RAM=20 bytes (so 2 small
    blocks fit), spillover=100, long=100."""
    spillover = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover.start()
    long_region = LongStorageRegion(tmp_path / "long", max_bytes=100)
    long_region.start()
    store = HostStore(max_bytes=20, spillover=spillover, long_region=long_region)
    return store, tmp_path


# ----------------------------------------------------------------------
# Backward compatibility: RAM-only store still works
# ----------------------------------------------------------------------


def test_ram_only_store_unchanged_when_no_ssd():
    """No SSD wired → behaves exactly like a RAM-only store."""
    store = HostStore(max_bytes=1024)
    accepted, _ = store.set(_key("a"), b"hello", retention="short")
    assert accepted
    assert store.get(_key("a")).value == b"hello"
    assert store.spillover_bytes == 0
    assert store.long_bytes == 0


# ----------------------------------------------------------------------
# Long retention: write_through to long region on SET
# ----------------------------------------------------------------------


def test_long_set_writes_to_long_region(store_with_ssd):
    """A retention=long SET should land in BOTH host RAM and the long SSD."""
    store, _ = store_with_ssd
    accepted, _ = store.set(_key("a"), b"x" * 10, retention="long")
    assert accepted

    # In RAM
    assert store.get(_key("a")).value == b"x" * 10
    # AND on long SSD (visible via the stat)
    assert store.long_bytes == 10
    # NOT on spillover
    assert store.spillover_bytes == 0


def test_long_set_recovers_after_ram_eviction(store_with_ssd):
    """Once a long block falls out of RAM (LRU when capacity is full),
    a subsequent GET should re-fetch from the long region and promote
    back to RAM."""
    store, _ = store_with_ssd
    store.set(_key("a"), b"x" * 10, retention="long")

    # Fill RAM with another long block — capacity is 20, so a+b = 20.
    store.set(_key("b"), b"y" * 10, retention="long")
    assert store.long_bytes == 20

    # Now insert a third long block — evicts the LRU long from RAM (but
    # the long region keeps it).
    store.set(_key("c"), b"z" * 10, retention="long")

    # GET 'a' must succeed even though it's not in RAM anymore.
    entry = store.get(_key("a"))
    assert entry is not None
    assert entry.value == b"x" * 10


# ----------------------------------------------------------------------
# Short retention: lazy write to spillover on RAM eviction
# ----------------------------------------------------------------------


def test_short_set_does_not_write_spillover_immediately(store_with_ssd):
    """retention=short — bytes stay in RAM, NO immediate disk write."""
    store, _ = store_with_ssd
    accepted, _ = store.set(_key("a"), b"hello", retention="short")
    assert accepted
    assert store.spillover_bytes == 0
    # But still readable from RAM.
    assert store.get(_key("a")).value == b"hello"


def test_short_eviction_writes_to_spillover(store_with_ssd):
    """Push enough short blocks through RAM to force an LRU eviction;
    the evicted block must land in spillover (lazy write_back)."""
    store, _ = store_with_ssd

    # Fill RAM. capacity = 20 bytes; 10-byte blocks → 2 fit.
    store.set(_key("a"), b"x" * 10, retention="short")
    store.set(_key("b"), b"y" * 10, retention="short")
    assert store.spillover_bytes == 0  # nothing evicted yet

    # Insert c — must evict the LRU short block (a) to make room.
    accepted, _ = store.set(_key("c"), b"z" * 10, retention="short")
    assert accepted
    # The evicted 'a' is now on spillover.
    assert store.spillover_bytes == 10

    # GET 'a' returns the spilled bytes (promoted back to RAM).
    entry = store.get(_key("a"))
    assert entry is not None
    assert entry.value == b"x" * 10


# ----------------------------------------------------------------------
# GET priority: RAM → long → spillover → miss
# ----------------------------------------------------------------------


def test_get_falls_through_all_tiers(store_with_ssd):
    """A key in spillover but not RAM/long is still reachable."""
    store, _ = store_with_ssd

    # Put a short block in RAM, then force eviction by filling.
    store.set(_key("a"), b"x" * 10, retention="short")
    store.set(_key("b"), b"y" * 10, retention="short")
    store.set(_key("c"), b"z" * 10, retention="short")  # evicts 'a' → spillover

    # 'a' lives in spillover only — GET should find it.
    assert store.get(_key("a")).value == b"x" * 10


def test_get_miss_returns_none_on_all_tiers_empty(store_with_ssd):
    store, _ = store_with_ssd
    assert store.get(_key("nope")) is None


# ----------------------------------------------------------------------
# Persistence: long region survives a store restart
# ----------------------------------------------------------------------


def test_long_blocks_survive_store_restart(tmp_path: Path):
    """The headline feature: kill+restart the daemon, long-retention
    blocks are still in the cache."""
    # Phase A: write a long block.
    spillover_a = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_a.start()
    long_a = LongStorageRegion(tmp_path / "long", max_bytes=100)
    long_a.start()
    store_a = HostStore(max_bytes=20, spillover=spillover_a, long_region=long_a)
    store_a.set(_key("durable"), b"important-data", retention="long", model="m1", compat_key="ck1")
    # Confirm it's there.
    assert store_a.get(_key("durable"), model="m1", compat_key="ck1").value == b"important-data"

    # Phase B: simulate restart by building a NEW store + regions
    # pointing at the same directories. Spillover wipes; long recovers.
    spillover_b = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_b.start()
    long_b = LongStorageRegion(tmp_path / "long", max_bytes=100)
    long_b.start()
    store_b = HostStore(max_bytes=20, spillover=spillover_b, long_region=long_b)

    # The block is gone from RAM (new process), but the long region
    # recovered the index from per-block .kvcache.metadata sidecars.
    entry = store_b.get(_key("durable"), model="m1", compat_key="ck1")
    assert entry is not None
    assert entry.value == b"important-data"


def test_short_blocks_do_NOT_survive_restart(tmp_path: Path):
    """Conversely: short-retention blocks land in spillover, which is
    wiped on restart. Persistence is a long-retention privilege."""
    spillover_a = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_a.start()
    long_a = LongStorageRegion(tmp_path / "long", max_bytes=100)
    long_a.start()
    store_a = HostStore(max_bytes=20, spillover=spillover_a, long_region=long_a)

    # Force a short eviction → goes to spillover.
    store_a.set(_key("a"), b"x" * 10, retention="short")
    store_a.set(_key("b"), b"y" * 10, retention="short")
    store_a.set(_key("c"), b"z" * 10, retention="short")  # evicts 'a' to spillover
    assert store_a.spillover_bytes > 0

    # Restart.
    spillover_b = SpilloverRegion(tmp_path / "spill", max_bytes=100)
    spillover_b.start()
    long_b = LongStorageRegion(tmp_path / "long", max_bytes=100)
    long_b.start()
    store_b = HostStore(max_bytes=20, spillover=spillover_b, long_region=long_b)

    # Spillover wiped — short blocks are gone.
    assert store_b.get(_key("a")) is None
    assert store_b.spillover_bytes == 0


# ----------------------------------------------------------------------
# exists() considers SSD tiers
# ----------------------------------------------------------------------


def test_exists_finds_blocks_in_ssd_tiers(store_with_ssd):
    """`exists` is used by routers for cache-locality decisions — it
    must consider SSD tiers, not just RAM."""
    store, _ = store_with_ssd

    # Put a long block; check it's findable via exists.
    store.set(_key("long_one"), b"l" * 5, retention="long")
    assert store.exists([_key("long_one")]) == [True]

    # Force a short block to spillover via eviction.
    store.set(_key("short_a"), b"x" * 10, retention="short")
    store.set(_key("short_b"), b"y" * 10, retention="short")
    # Now RAM has short_a, short_b. Insert c → evict short_a → spillover.
    # But wait — RAM has 20 bytes capacity, we have 10+10=20. Plus long
    # one was 5 in RAM. So RAM is over capacity → some eviction already.
    # The exists() must find blocks regardless of which tier.

    # All three should be reported present.
    present = store.exists([_key("long_one"), _key("short_a"), _key("short_b")])
    assert all(present)


# ----------------------------------------------------------------------
# clear() drops SSD entries
# ----------------------------------------------------------------------


def test_clear_drops_ssd_entries_too(store_with_ssd):
    store, _ = store_with_ssd
    store.set(_key("a"), b"x" * 10, retention="long")
    store.set(_key("b"), b"y" * 10, retention="short")
    store.set(_key("c"), b"z" * 10, retention="short")  # evicts b → spillover

    assert store.long_bytes > 0
    assert store.spillover_bytes > 0

    store.clear()

    assert store.long_bytes == 0
    assert store.spillover_bytes == 0
    assert store.get(_key("a")) is None
    assert store.get(_key("b")) is None


# ----------------------------------------------------------------------
# get_many — batched multi-key read (BatchGet path)
# ----------------------------------------------------------------------


def test_get_many_order_and_miss_fallback(store_with_ssd):
    """LongStorageRegion has no get_bytes_batch → get_many uses the per-key
    fallback. Verify order preservation + a miss in the middle."""
    store, _ = store_with_ssd
    store.set(_key("a"), b"AAAA", retention="long", model="m", compat_key="c")
    store.set(_key("b"), b"BBBB", retention="long", model="m", compat_key="c")
    out = store.get_many([_key("b"), _key("zzz"), _key("a")], model="m", compat_key="c")
    assert [bytes(o.value) if o is not None else None for o in out] == [
        b"BBBB",
        None,
        b"AAAA",
    ]


class _CountingLongRegion(LongStorageRegion):
    """LongStorageRegion + a counting get_bytes_batch, to prove get_many
    reads the long region in ONE batch call rather than N single gets."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.batch_calls = 0
        self.single_get_calls = 0

    def get_bytes_batch(self, keys, *, model="", compat_key=""):
        self.batch_calls += 1
        return [
            super(_CountingLongRegion, self).get_bytes(k, model=model, compat_key=compat_key)
            for k in keys
        ]

    def get_bytes(self, key, *, model="", compat_key=""):
        self.single_get_calls += 1
        return super().get_bytes(key, model=model, compat_key=compat_key)


def test_get_many_uses_one_batch_call(tmp_path: Path):
    spill = SpilloverRegion(tmp_path / "s", max_bytes=1000)
    spill.start()
    longr = _CountingLongRegion(tmp_path / "l", max_bytes=1000)
    longr.start()
    store = HostStore(max_bytes=8, spillover=spill, long_region=longr)  # RAM ~2 blocks
    for i in range(4):
        store.set(_key(f"k{i}"), f"v{i}".encode(), retention="long", model="m", compat_key="c")
    longr.batch_calls = 0
    longr.single_get_calls = 0

    out = store.get_many(
        [_key(f"k{i}") for i in range(4)] + [_key("miss")], model="m", compat_key="c"
    )
    assert [bytes(o.value) if o is not None else None for o in out] == [
        b"v0",
        b"v1",
        b"v2",
        b"v3",
        None,
    ]
    # The RAM-missed keys were read from the long region in a SINGLE batch
    # call — not per-key get_bytes.
    assert longr.batch_calls == 1
    assert longr.single_get_calls == 0
