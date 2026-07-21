###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/tablespace_multipool.py.

Validates:
- routing: smallest pool that fits wins
- get fall-through across pools
- exists OR across pools
- migration: same key written with a smaller value moves to a smaller pool
- restart-survival per-pool
- pool spec parser for CLI
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infera.kvd.tablespace_multipool import (
    MultiPoolTablespaceLongRegion,
    _format_slot_label,
    _parse_size_label,
    parse_pools_spec,
)


def _make_region(tmp_path: Path, **kw) -> MultiPoolTablespaceLongRegion:
    """Tiny pools for unit tests: 8 KB + 32 KB + 128 KB slots."""
    return MultiPoolTablespaceLongRegion(
        path=tmp_path,
        pools=kw.pop(
            "pools",
            [
                (8 * 1024, 64 * 1024),  # 8K slots, 64 KB total
                (32 * 1024, 256 * 1024),  # 32K slots, 256 KB total
                (128 * 1024, 512 * 1024),  # 128K slots, 512 KB total
            ],
        ),
        container_bytes=kw.pop("container_bytes", 64 * 1024),
        sync_writes=kw.pop("sync_writes", False),
        o_direct=kw.pop("o_direct", False),
    )


# ----------------------------------------------------------------------
# Basic put/get
# ----------------------------------------------------------------------


def test_put_routes_to_smallest_fitting_pool(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        # 100-byte value → routes to 8K pool (smallest that fits)
        ok, reason = r.put(b"small", b"x" * 100, retention="long", model="m", compat_key="c")
        assert ok, reason
        e = r.get_entry(b"small", model="m", compat_key="c")
        assert e is not None
        # 8K pool dir name in path
        # (entry doesn't expose pool; verify indirectly via used_bytes)
        # The 8K-slot pool has used_bytes==100; others 0.
        used_by_slot = {p.slot_bytes: p.used_bytes for p in r._pools}
        assert used_by_slot[8 * 1024] == 100
        assert used_by_slot[32 * 1024] == 0
        assert used_by_slot[128 * 1024] == 0
    finally:
        r.shutdown()


def test_put_routes_medium_value_to_32k_pool(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        # 20 KB value: doesn't fit in 8K, fits in 32K, chooses 32K (not 128K).
        ok, _ = r.put(b"medium", b"y" * (20 * 1024), retention="long", model="m", compat_key="c")
        assert ok
        used_by_slot = {p.slot_bytes: p.used_bytes for p in r._pools}
        assert used_by_slot[8 * 1024] == 0
        assert used_by_slot[32 * 1024] == 20 * 1024
        assert used_by_slot[128 * 1024] == 0
    finally:
        r.shutdown()


def test_put_routes_large_value_to_128k_pool(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, _ = r.put(b"large", b"z" * (100 * 1024), retention="long", model="m", compat_key="c")
        assert ok
        used_by_slot = {p.slot_bytes: p.used_bytes for p in r._pools}
        assert used_by_slot[8 * 1024] == 0
        assert used_by_slot[32 * 1024] == 0
        assert used_by_slot[128 * 1024] == 100 * 1024
    finally:
        r.shutdown()


def test_put_rejects_value_larger_than_largest_pool(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        ok, reason = r.put(
            b"huge", b"!" * (200 * 1024), retention="long", model="m", compat_key="c"
        )
        assert ok is False
        assert "exceeds_largest_pool" in (reason or "")
    finally:
        r.shutdown()


def test_get_falls_through_pools(tmp_path: Path):
    """Keys distributed across pools all reachable via the multi-pool
    get_bytes — no need to know which pool a key lives in."""
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k-small", b"a" * 100, retention="long", model="m", compat_key="c")
        r.put(b"k-medium", b"b" * 20480, retention="long", model="m", compat_key="c")
        r.put(b"k-large", b"c" * 102400, retention="long", model="m", compat_key="c")

        assert r.get_bytes(b"k-small", model="m", compat_key="c") == b"a" * 100
        assert r.get_bytes(b"k-medium", model="m", compat_key="c") == b"b" * 20480
        assert r.get_bytes(b"k-large", model="m", compat_key="c") == b"c" * 102400
        assert r.get_bytes(b"k-absent", model="m", compat_key="c") is None
    finally:
        r.shutdown()


def test_exists_returns_or_across_pools(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"a", b"x" * 100, retention="long", model="m", compat_key="c")
        r.put(b"b", b"y" * 20480, retention="long", model="m", compat_key="c")
        result = r.exists([b"a", b"b", b"missing"], model="m", compat_key="c")
        assert result == [True, True, False]
    finally:
        r.shutdown()


def test_exists_empty_input_returns_empty(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        assert r.exists([], model="m", compat_key="c") == []
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Cross-pool migration (same key, value shrinks)
# ----------------------------------------------------------------------


def test_same_key_value_shrinking_migrates_to_smaller_pool(tmp_path: Path):
    """A second PUT with a smaller value should move the key into the
    smaller-slot pool — and remove it from the old pool — so a future
    get returns the new value, not the stale one."""
    r = _make_region(tmp_path)
    r.start()
    try:
        # First write: 20 KB → 32K pool
        r.put(b"k", b"BIG" * 7000, retention="long", model="m", compat_key="c")
        used_first = {p.slot_bytes: p.used_bytes for p in r._pools}
        assert used_first[32 * 1024] > 0

        # Update with smaller value: 100 bytes → 8K pool
        r.put(b"k", b"tiny", retention="long", model="m", compat_key="c")

        used_second = {p.slot_bytes: p.used_bytes for p in r._pools}
        # 32K pool dropped its entry; 8K pool has the new one.
        assert used_second[32 * 1024] == 0
        assert used_second[8 * 1024] == 4  # len(b"tiny")

        # get returns the NEW value, not the stale one.
        assert r.get_bytes(b"k", model="m", compat_key="c") == b"tiny"
    finally:
        r.shutdown()


def test_same_key_value_growing_migrates_to_bigger_pool(tmp_path: Path):
    """Conversely: value grows past current pool's slot → migrate up."""
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k", b"small", retention="long", model="m", compat_key="c")
        used_first = {p.slot_bytes: p.used_bytes for p in r._pools}
        assert used_first[8 * 1024] > 0

        # Now write a 50K value — doesn't fit in 8K, must go to 128K.
        r.put(b"k", b"X" * (50 * 1024), retention="long", model="m", compat_key="c")

        used_second = {p.slot_bytes: p.used_bytes for p in r._pools}
        # 8K pool no longer holds it.
        assert used_second[8 * 1024] == 0
        # 128K pool has it (50K doesn't fit in 32K either).
        assert used_second[128 * 1024] == 50 * 1024
        assert r.get_bytes(b"k", model="m", compat_key="c") == b"X" * (50 * 1024)
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Restart survival per-pool
# ----------------------------------------------------------------------


def test_restart_preserves_entries_in_all_pools(tmp_path: Path):
    """Each pool independently snapshots; restart-survival is per-pool."""
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"small", b"s" * 100, retention="long", model="m", compat_key="c")
    r1.put(b"med", b"m" * 20480, retention="long", model="m", compat_key="c")
    r1.put(b"big", b"b" * 102400, retention="long", model="m", compat_key="c")
    r1.shutdown()

    r2 = _make_region(tmp_path)
    r2.start()
    try:
        assert r2.entries_count == 3
        assert r2.get_bytes(b"small", model="m", compat_key="c") == b"s" * 100
        assert r2.get_bytes(b"med", model="m", compat_key="c") == b"m" * 20480
        assert r2.get_bytes(b"big", model="m", compat_key="c") == b"b" * 102400
    finally:
        r2.shutdown()


def test_one_pool_geometry_change_doesnt_break_others(tmp_path: Path):
    """Switching slot_bytes on one pool ignores its snapshot but the
    others still recover. Operator changing config mid-deploy shouldn't
    lose ALL data."""
    r1 = _make_region(tmp_path)
    r1.start()
    r1.put(b"in-small", b"a" * 100, retention="long", model="m", compat_key="c")
    r1.put(b"in-big", b"b" * 50000, retention="long", model="m", compat_key="c")
    r1.shutdown()

    # Reopen with DIFFERENT slot_bytes for the smallest pool.
    r2 = MultiPoolTablespaceLongRegion(
        path=tmp_path,
        pools=[
            (16 * 1024, 64 * 1024),  # ← was 8K, now 16K
            (32 * 1024, 256 * 1024),
            (128 * 1024, 512 * 1024),
        ],
        container_bytes=64 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    r2.start()
    try:
        # The 16K pool is a fresh directory (it didn't exist in r1) →
        # `in-small` is lost. But `in-big` (in the 128K pool) survives.
        # NB: the new 16K pool has its own pool-XXXX dir; the OLD 8K
        # pool's directory `pool-0000008K` is now orphaned on disk
        # (still there, GC would prune in real ops).
        assert r2.get_bytes(b"in-small", model="m", compat_key="c") is None
        assert r2.get_bytes(b"in-big", model="m", compat_key="c") == b"b" * 50000
    finally:
        r2.shutdown()


# ----------------------------------------------------------------------
# Remove / clear
# ----------------------------------------------------------------------


def test_remove_works_across_pools(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"k-small", b"s" * 100, retention="long", model="m", compat_key="c")
        r.put(b"k-big", b"b" * 50000, retention="long", model="m", compat_key="c")

        removed_small = r.remove(b"k-small", model="m", compat_key="c")
        assert removed_small is True
        removed_big = r.remove(b"k-big", model="m", compat_key="c")
        assert removed_big is True
        assert r.entries_count == 0
        removed_missing = r.remove(b"k-missing", model="m", compat_key="c")
        assert removed_missing is False
    finally:
        r.shutdown()


def test_clear_drops_all_pools(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        for i in range(3):
            r.put(f"a{i}".encode(), b"a" * 100, retention="long", model="m", compat_key="c")
            r.put(f"b{i}".encode(), b"b" * 30000, retention="long", model="m", compat_key="c")
        assert r.entries_count == 6

        total = r.clear()
        assert total == 6
        assert r.entries_count == 0
        assert r.used_bytes == 0
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Properties — sum across pools
# ----------------------------------------------------------------------


def test_max_bytes_is_sum_across_pools(tmp_path: Path):
    r = _make_region(tmp_path)
    # 64K + 256K + 512K = 832K
    assert r.max_bytes == 64 * 1024 + 256 * 1024 + 512 * 1024


def test_used_bytes_is_sum_across_pools(tmp_path: Path):
    r = _make_region(tmp_path)
    r.start()
    try:
        r.put(b"a", b"x" * 100, retention="long", model="m", compat_key="c")
        r.put(b"b", b"y" * 20480, retention="long", model="m", compat_key="c")
        # 100 + 20480 = 20580
        assert r.used_bytes == 20580
    finally:
        r.shutdown()


def test_pool_slot_sizes_property(tmp_path: Path):
    r = _make_region(tmp_path)
    assert r.pool_slot_sizes == (8 * 1024, 32 * 1024, 128 * 1024)


# ----------------------------------------------------------------------
# Construction / spec parsing
# ----------------------------------------------------------------------


def test_construction_rejects_empty_pools_list(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one pool"):
        MultiPoolTablespaceLongRegion(
            path=tmp_path, pools=[], container_bytes=4 * 1024, o_direct=False
        )


def test_construction_rejects_duplicate_slot_bytes(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate"):
        MultiPoolTablespaceLongRegion(
            path=tmp_path,
            pools=[(8 * 1024, 64 * 1024), (8 * 1024, 128 * 1024)],
            container_bytes=4 * 1024,
            o_direct=False,
        )


def test_construction_sorts_pools_by_slot_size(tmp_path: Path):
    """Operator passes pools in any order; we always route smallest-first."""
    r = MultiPoolTablespaceLongRegion(
        path=tmp_path,
        pools=[
            (128 * 1024, 512 * 1024),  # bigger first in spec
            (8 * 1024, 64 * 1024),
            (32 * 1024, 256 * 1024),
        ],
        container_bytes=64 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    assert r.pool_slot_sizes == (8 * 1024, 32 * 1024, 128 * 1024)


def test_format_slot_label_is_sortable():
    """Directory names should sort lexically the same as slot sizes
    sort numerically — easy ls."""
    labels = [
        _format_slot_label(8 * 1024),
        _format_slot_label(64 * 1024),
        _format_slot_label(512 * 1024),
        _format_slot_label(1024 * 1024),
        _format_slot_label(4 * 1024 * 1024),
    ]
    assert labels == sorted(labels)


def test_parse_size_label():
    assert _parse_size_label("64K") == 64 * 1024
    assert _parse_size_label("1M") == 1024 * 1024
    assert _parse_size_label("4G") == 4 * 1024**3
    assert _parse_size_label("4096") == 4096


def test_parse_pools_spec_equal_share():
    pools = parse_pools_spec("64K,1M", default_max_bytes_per_pool=8 * 1024**3)
    assert len(pools) == 2
    sizes = [p[0] for p in pools]
    assert sizes == [64 * 1024, 1024 * 1024]
    # Equal split → each pool gets default
    assert pools[0][1] == 8 * 1024**3
    assert pools[1][1] == 8 * 1024**3


def test_parse_pools_spec_weighted():
    pools = parse_pools_spec("64K*1,1M*4,4M*1", default_max_bytes_per_pool=8 * 1024**3)
    # mean share = (1+4+1)/3 = 2
    # 64K gets 1/2 × default = 4 GB
    # 1M gets 4/2 × default = 16 GB
    # 4M gets 1/2 × default = 4 GB
    assert pools[0] == (64 * 1024, 4 * 1024**3)
    assert pools[1] == (1024 * 1024, 16 * 1024**3)
    assert pools[2] == (4 * 1024**2, 4 * 1024**3)


def test_parse_pools_spec_single_pool():
    pools = parse_pools_spec("64K", default_max_bytes_per_pool=8 * 1024**3)
    assert pools == [(64 * 1024, 8 * 1024**3)]


def test_parse_pools_spec_rejects_empty():
    with pytest.raises(ValueError):
        parse_pools_spec("", default_max_bytes_per_pool=1)


def test_clear_namespace_removes_matching_entries_across_pools(tmp_path: Path):
    """HostStore.clear(model=..., compat_key=...)
    previously bypassed multipool because the wrapper has no `_entries`
    attribute. The new `clear_namespace` method walks every pool and
    removes matching entries. Verify cross-pool behavior: put one
    block in the 8 KB pool and one in the 32 KB pool under the same
    (model, compat_key); clear_namespace should remove both."""
    region = MultiPoolTablespaceLongRegion(
        path=tmp_path,
        pools=[(8 * 1024, 256 * 1024), (32 * 1024, 256 * 1024)],
        container_bytes=64 * 1024,
        sync_writes=False,
        o_direct=False,
    )
    region.start()
    try:
        # Small value lands in the 8K pool, larger in the 32K pool.
        ok_a, _ = region.put(
            b"k_small",
            b"x" * 100,
            retention="long",
            model="m1",
            compat_key="ck1",
        )
        ok_b, _ = region.put(
            b"k_big",
            b"y" * 20_000,
            retention="long",
            model="m1",
            compat_key="ck1",
        )
        # Another (model, compat_key) — should survive the clear.
        ok_c, _ = region.put(
            b"k_other",
            b"z" * 100,
            retention="long",
            model="m2",
            compat_key="ck2",
        )
        assert ok_a and ok_b and ok_c
        assert region.entries_count == 3

        removed = region.clear_namespace("m1", "ck1")
        assert removed == 2
        assert region.entries_count == 1
        assert region.get_bytes(b"k_other", model="m2", compat_key="ck2") == b"z" * 100
        assert region.get_bytes(b"k_small", model="m1", compat_key="ck1") is None
        assert region.get_bytes(b"k_big", model="m1", compat_key="ck1") is None
    finally:
        region.shutdown()
