###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Integration: HostStore + StripedLongRegion.

StripedLongRegion is a drop-in for TablespaceLongRegion at the
HostStore level — same put/get/exists/clear surface. This file
exercises the standard HostStore flow with a 4-shard striped region
underneath and asserts results match a single-shard baseline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infera.kvd.ssd import SpilloverRegion
from infera.kvd.store import HostStore
from infera.kvd.striped_long_region import StripedLongRegion
from infera.kvd.tablespace import TablespaceLongRegion


def _make_shards(root: Path, n: int) -> list[TablespaceLongRegion]:
    shards = []
    for i in range(n):
        path = root / f"shard{i}"
        path.mkdir(parents=True, exist_ok=True)
        shards.append(
            TablespaceLongRegion(
                path,
                max_bytes=128,
                slot_bytes=16,
                container_bytes=64,
                sync_writes=False,
                o_direct=False,
            )
        )
    return shards


@pytest.fixture
def store_with_striped(tmp_path: Path):
    spillover = SpilloverRegion(tmp_path / "spill", max_bytes=200)
    spillover.start()
    long_region = StripedLongRegion(_make_shards(tmp_path / "long", n=4))
    long_region.start()
    store = HostStore(max_bytes=20, spillover=spillover, long_region=long_region)
    yield store, tmp_path, long_region
    long_region.shutdown()
    spillover.shutdown()


def _key(s: str) -> bytes:
    # Pad to 8 bytes so the slot_bytes=16 region accepts the key (value
    # padding is what matters; key length is independent).
    return s.encode("ascii").ljust(8, b"\x00")


# ----------------------------------------------------------------------
# Long SET write-through to striped region
# ----------------------------------------------------------------------


def test_long_set_writes_to_striped_region(store_with_striped):
    store, _, long_region = store_with_striped
    ok, reason = store.set(_key("blockA"), b"v" * 10, retention="long", model="m", compat_key="ck")
    assert ok, reason
    assert long_region.entries_count == 1
    assert long_region.used_bytes == 10


def test_long_block_recoverable_after_ram_eviction(store_with_striped):
    """Same scenario as test_store_with_tablespace's eviction test:
    fill RAM beyond budget, long blocks survive on the striped region."""
    store, _, _ = store_with_striped
    # 20-byte RAM budget, 10-byte values → 2 fit. Third forces eviction.
    for i in range(3):
        ok, _ = store.set(
            _key(f"k{i}"),
            b"v" * 10,
            retention="long",
            model="m",
            compat_key="ck",
        )
        assert ok
    # All three are findable through HostStore.get (which promotes
    # back from the striped region).
    for i in range(3):
        entry = store.get(_key(f"k{i}"), model="m", compat_key="ck")
        assert entry is not None, i
        assert store.resolve_value(entry) == b"v" * 10


def test_exists_through_host_store_finds_striped_entries(store_with_striped):
    store, _, _ = store_with_striped
    for i in range(8):
        ok, _ = store.set(
            _key(f"k{i}"),
            b"v" * 8,
            retention="long",
            model="m",
            compat_key="ck",
        )
        assert ok
    keys = [_key(f"k{i}") for i in range(8)] + [_key(f"absent{i}") for i in range(4)]
    res = store.exists(keys, model="m", compat_key="ck")
    assert res == [True] * 8 + [False] * 4


def test_clear_drops_striped_entries(store_with_striped):
    store, _, long_region = store_with_striped
    for i in range(8):
        ok, _ = store.set(
            _key(f"k{i}"),
            b"v" * 8,
            retention="long",
            model="m",
            compat_key="ck",
        )
        assert ok
    assert long_region.entries_count == 8
    # HostStore.clear() with no namespace clears the lot.
    store.clear()
    assert long_region.entries_count == 0


def test_results_match_single_shard_baseline(tmp_path: Path):
    """The canonical test: drive the same workload through a single-shard
    HostStore and a 4-shard HostStore and assert the externally visible
    answers are identical."""
    # Build twin stores.
    spillover_a = SpilloverRegion(tmp_path / "spill-a", max_bytes=200)
    spillover_a.start()
    long_a = TablespaceLongRegion(
        tmp_path / "long-a",
        max_bytes=4 * 128,
        slot_bytes=16,
        container_bytes=64,
        sync_writes=False,
        o_direct=False,
    )
    long_a.start()
    store_a = HostStore(max_bytes=20, spillover=spillover_a, long_region=long_a)

    spillover_b = SpilloverRegion(tmp_path / "spill-b", max_bytes=200)
    spillover_b.start()
    long_b = StripedLongRegion(_make_shards(tmp_path / "long-b", n=4))
    long_b.start()
    store_b = HostStore(max_bytes=20, spillover=spillover_b, long_region=long_b)

    try:
        n = 16
        keys = [_key(f"k{i}") for i in range(n)]
        values = [f"v{i}".encode() + b"x" * 4 for i in range(n)]
        for k, v in zip(keys, values, strict=True):
            ok_a, _ = store_a.set(k, v, retention="long", model="m", compat_key="ck")
            ok_b, _ = store_b.set(k, v, retention="long", model="m", compat_key="ck")
            assert ok_a == ok_b is True

        # exists must agree.
        all_keys = keys + [_key(f"absent{i}") for i in range(4)]
        assert store_a.exists(all_keys, model="m", compat_key="ck") == store_b.exists(
            all_keys, model="m", compat_key="ck"
        )

        # get must agree (modulo resolve_value to unwrap arena).
        for k in all_keys:
            ea = store_a.get(k, model="m", compat_key="ck")
            eb = store_b.get(k, model="m", compat_key="ck")
            if ea is None:
                assert eb is None
            else:
                assert eb is not None
                assert store_a.resolve_value(ea) == store_b.resolve_value(eb)

        # Both report the SAME entries_count on the long region.
        assert long_a.entries_count == long_b.entries_count == n
    finally:
        long_a.shutdown()
        long_b.shutdown()
        spillover_a.shutdown()
        spillover_b.shutdown()
