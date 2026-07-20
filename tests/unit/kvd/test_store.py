###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/store.py — priority-aware host RAM LRU."""

from __future__ import annotations

import time

import pytest

from infera.kvd.store import HostStore, retention_priority


def _key(s: str) -> bytes:
    """Convenience: make a fixed-length 8-byte key from a short string."""
    return s.encode("ascii").ljust(8, b"\x00")


# ----------------------------------------------------------------------
# Basic set/get/exists
# ----------------------------------------------------------------------


def test_set_and_get_round_trip():
    store = HostStore(max_bytes=1024)
    accepted, reason = store.set(_key("a"), b"hello", retention="short")
    assert accepted is True
    assert reason is None
    entry = store.get(_key("a"))
    assert entry is not None
    assert entry.value == b"hello"
    assert entry.retention == "short"


def test_get_miss_returns_none():
    store = HostStore(max_bytes=1024)
    assert store.get(_key("nope")) is None


def test_exists_bulk():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"x", retention="short")
    store.set(_key("c"), b"y", retention="long")
    present = store.exists([_key("a"), _key("b"), _key("c")])
    assert present == [True, False, True]


def test_exists_empty_keys_returns_empty():
    store = HostStore(max_bytes=1024)
    assert store.exists([]) == []


def test_set_invalid_retention_raises():
    store = HostStore(max_bytes=1024)
    with pytest.raises(ValueError, match="unknown retention"):
        store.set(_key("a"), b"x", retention="forever")


def test_zero_max_bytes_raises():
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        HostStore(max_bytes=0)


# ----------------------------------------------------------------------
# Namespacing by (model, compat_key)
# ----------------------------------------------------------------------


def test_same_key_different_model_does_not_collide():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"value-A", retention="short", model="modelA")
    store.set(_key("a"), b"value-B", retention="short", model="modelB")
    assert store.get(_key("a"), model="modelA").value == b"value-A"
    assert store.get(_key("a"), model="modelB").value == b"value-B"


def test_compat_key_separates_quant_variants():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"fp16", retention="short", model="m", compat_key="fp16")
    store.set(_key("a"), b"fp8", retention="short", model="m", compat_key="fp8")
    assert store.get(_key("a"), model="m", compat_key="fp16").value == b"fp16"
    assert store.get(_key("a"), model="m", compat_key="fp8").value == b"fp8"


# ----------------------------------------------------------------------
# Priority-aware eviction
# ----------------------------------------------------------------------


def test_short_block_evicts_short_block_under_pressure():
    """Equal priority — LRU within the priority class kicks in."""
    store = HostStore(max_bytes=20)  # holds two 10-byte blocks
    a, b, c = b"x" * 10, b"y" * 10, b"z" * 10
    store.set(_key("a"), a, retention="short")
    time.sleep(0.01)  # make sure timestamps differ
    store.set(_key("b"), b, retention="short")
    # Both fit. Now insert c — must evict the LRU (a).
    accepted, _ = store.set(_key("c"), c, retention="short")
    assert accepted
    assert store.get(_key("a")) is None  # evicted
    assert store.get(_key("b")) is not None
    assert store.get(_key("c")) is not None


def test_short_block_cannot_evict_long_block():
    """A short-retention SET must NOT displace a long-retention entry.
    The store rejects the SET; caller decides what to do."""
    store = HostStore(max_bytes=10)
    store.set(_key("long"), b"x" * 10, retention="long")
    accepted, reason = store.set(_key("short"), b"y" * 10, retention="short")
    assert accepted is False
    assert reason == "would_displace_higher_priority"
    # `long` block is still there.
    assert store.get(_key("long")) is not None


def test_long_block_can_evict_short_block():
    """A long-retention SET DOES displace lower-priority entries."""
    store = HostStore(max_bytes=10)
    store.set(_key("short"), b"x" * 10, retention="short")
    accepted, _ = store.set(_key("long"), b"y" * 10, retention="long")
    assert accepted
    assert store.get(_key("short")) is None  # evicted
    assert store.get(_key("long")) is not None


def test_none_retention_evicted_before_short():
    """Priority ordering: NONE < SHORT < LONG."""
    store = HostStore(max_bytes=20)
    store.set(_key("none"), b"x" * 10, retention="none")
    store.set(_key("short"), b"y" * 10, retention="short")
    accepted, _ = store.set(_key("new"), b"z" * 10, retention="short")
    assert accepted
    assert store.get(_key("none")) is None
    assert store.get(_key("short")) is not None
    assert store.get(_key("new")) is not None


# ----------------------------------------------------------------------
# Update semantics
# ----------------------------------------------------------------------


def test_set_existing_key_updates_value():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"v1", retention="short")
    accepted, _ = store.set(_key("a"), b"v2-longer", retention="short")
    assert accepted
    assert store.get(_key("a")).value == b"v2-longer"


def test_set_upgrades_retention_short_to_long():
    """Retention upgrades are allowed (same-key, higher-priority)."""
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"v", retention="short")
    accepted, _ = store.set(_key("a"), b"v", retention="long")
    assert accepted
    assert store.get(_key("a")).retention == "long"


def test_set_blocks_retention_downgrade():
    """Downgrade long → short would let a misbehaving client demote
    pinned content. Refused."""
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"v", retention="long")
    accepted, reason = store.set(_key("a"), b"v", retention="short")
    assert accepted is False
    assert reason == "retention_downgrade_not_allowed"
    # Original retention preserved.
    assert store.get(_key("a")).retention == "long"


def test_set_value_larger_than_store_rejected():
    store = HostStore(max_bytes=10)
    accepted, reason = store.set(_key("big"), b"x" * 100, retention="short")
    assert accepted is False
    assert reason == "value_larger_than_store"


# ----------------------------------------------------------------------
# Clear semantics
# ----------------------------------------------------------------------


def test_clear_all_drops_everything():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"x", retention="short", model="m1")
    store.set(_key("b"), b"y", retention="long", model="m2")
    count = store.clear()
    assert count == 2
    assert store.get(_key("a"), model="m1") is None
    assert store.get(_key("b"), model="m2") is None


def test_clear_by_model_keeps_others():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"x", retention="short", model="m1")
    store.set(_key("b"), b"y", retention="long", model="m2")
    count = store.clear(model="m1")
    assert count == 1
    assert store.get(_key("a"), model="m1") is None
    assert store.get(_key("b"), model="m2") is not None


def test_clear_unknown_model_is_zero():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"x", retention="short", model="m1")
    assert store.clear(model="nope") == 0
    assert store.get(_key("a"), model="m1") is not None


# ----------------------------------------------------------------------
# Stats counters
# ----------------------------------------------------------------------


def test_stats_counts_gets_and_hits_misses():
    store = HostStore(max_bytes=1024)
    store.set(_key("a"), b"x", retention="short")
    store.get(_key("a"))  # hit
    store.get(_key("a"))  # hit
    store.get(_key("nope"))  # miss
    s = store.stats
    assert s.entries == 1
    assert s.host_bytes == 1
    assert s.gets_total == 3
    assert s.hits_total == 2
    assert s.misses_total == 1
    assert s.sets_total == 1


def test_stats_evictions_counted():
    store = HostStore(max_bytes=10)
    store.set(_key("a"), b"x" * 10, retention="short")
    store.set(_key("b"), b"y" * 10, retention="short")  # evicts a
    assert store.stats.evictions_total == 1


def test_stats_rejections_counted():
    store = HostStore(max_bytes=10)
    store.set(_key("long"), b"x" * 10, retention="long")
    store.set(_key("short"), b"y" * 10, retention="short")  # rejected
    assert store.stats.set_rejections_total == 1
    assert store.stats.evictions_total == 0


# ----------------------------------------------------------------------
# Priority helper
# ----------------------------------------------------------------------


def test_retention_priority_ordering():
    assert retention_priority("none") < retention_priority("short")
    assert retention_priority("short") < retention_priority("long")


def test_retention_priority_unknown_raises():
    with pytest.raises(ValueError):
        retention_priority("forever")


# ----------------------------------------------------------------------
# Concurrent-PUT RAM↔SSD coherence
# ----------------------------------------------------------------------


class _RecordingLongRegion:
    """Drop-in replacement for `LongStorageRegion` that records put
    order. Locks while writing to simulate real SSD latency, which is
    what makes the race observable."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.puts: list[tuple[bytes, bytes]] = []  # (key, value) in put order
        self._values: dict[bytes, bytes] = {}

    def put(self, key, value, *, retention, model, compat_key, metadata):
        # Simulate ~1 ms of disk latency to widen the race window.
        import time

        time.sleep(0.001)
        with self._lock:
            self.puts.append((key, value))
            self._values[key] = value
        return True, None

    def value_of(self, key: bytes) -> bytes | None:
        return self._values.get(key)


def test_concurrent_set_same_key_ram_and_long_region_converge():
    """Two threads SET the same key with different values. After both
    return, RAM and the long region MUST hold the same bytes. Without
    `_long_write_lock` + re-read-ram-under-lock, the previous code
    could leave RAM=B but SSD=A.
    """
    import threading

    long_region = _RecordingLongRegion()
    store = HostStore(max_bytes=4096, long_region=long_region)

    key = _key("same")
    payload_a = b"A" * 64
    payload_b = b"B" * 64
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def _writer(payload: bytes) -> None:
        barrier.wait()  # release both threads at the same time
        ok, _ = store.set(key, payload, retention="long")
        results.append(ok)

    t_a = threading.Thread(target=_writer, args=(payload_a,))
    t_b = threading.Thread(target=_writer, args=(payload_b,))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert results == [True, True]
    # RAM holds one of the two values.
    ram_entry = store.get(key)
    assert ram_entry is not None
    ram_val = ram_entry.value
    assert ram_val in (payload_a, payload_b)
    # SSD holds the SAME value as RAM — the convergence guarantee.
    ssd_val = long_region.value_of(key)
    assert ssd_val == ram_val, (
        f"RAM={ram_val!r} != SSD={ssd_val!r} — concurrent SETs left "
        f"RAM and SSD inconsistent; the put-order in long_region was "
        f"{[v for _, v in long_region.puts]}"
    )


def test_concurrent_set_runs_serialized_via_long_write_lock(monkeypatch):
    """`_long_write_lock` should serialize long-region writes — verified
    by observing that no two writes can overlap. Uses a long-region stub
    that records (start, end) timestamps and asserts non-overlap."""
    import threading
    import time

    class _OverlapDetector:
        def __init__(self):
            self._lock = threading.Lock()
            self._in_flight = 0
            self.max_concurrent_observed = 0
            self.calls = 0

        def put(self, key, value, *, retention, model, compat_key, metadata):
            with self._lock:
                self._in_flight += 1
                self.max_concurrent_observed = max(self.max_concurrent_observed, self._in_flight)
                self.calls += 1
            time.sleep(0.005)  # widen race window
            with self._lock:
                self._in_flight -= 1
            return True, None

    long_region = _OverlapDetector()
    store = HostStore(max_bytes=4096, long_region=long_region)
    threads = [
        threading.Thread(
            target=store.set,
            args=(_key(f"k{i}"), b"x" * 16),
            kwargs={"retention": "long"},
        )
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert long_region.calls == 8
    # With _long_write_lock, max concurrency is 1 (serialized).
    assert long_region.max_concurrent_observed == 1
