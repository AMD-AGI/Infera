###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera.kvd.striped_long_region.

Covers the StripedLongRegion wrapper around N TablespaceLongRegion
shards: deterministic hash routing, fanout via ThreadPoolExecutor,
single-key delegation, multi-key batched reads with order preservation
+ misses, lifecycle, stats aggregation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infera.kvd.striped_long_region import (
    StripedLongRegion,
    composite_bytes,
)
from infera.kvd.tablespace import TablespaceLongRegion


def _make_shard(path: Path, **kw) -> TablespaceLongRegion:
    """Tiny per-shard region for unit tests. 8 KB slots × 4 slots ×
    4 containers = 128 KB total per shard. Buffered IO so we can
    write at any 8 KB slot size without 4 KB alignment fights."""
    path.mkdir(parents=True, exist_ok=True)
    return TablespaceLongRegion(
        path=path,
        max_bytes=kw.pop("max_bytes", 128 * 1024),
        slot_bytes=kw.pop("slot_bytes", 8 * 1024),
        container_bytes=kw.pop("container_bytes", 32 * 1024),
        sync_writes=kw.pop("sync_writes", False),
        o_direct=kw.pop("o_direct", False),
    )


def _make_striped(tmp_path: Path, n: int = 4, **shard_kw) -> StripedLongRegion:
    shards = [_make_shard(tmp_path / f"shard{i}", **shard_kw) for i in range(n)]
    region = StripedLongRegion(shards)
    region.start()
    return region


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_construction_rejects_empty_shards():
    with pytest.raises(ValueError, match="at least one shard"):
        StripedLongRegion([])


def test_composite_bytes_is_distinct_across_tenants():
    """Same key bytes in different (model, compat_key) namespaces
    must map to different shard-routing inputs — otherwise tenants
    would collide on shard pinning."""
    a = composite_bytes("m1", "ck1", b"key")
    b = composite_bytes("m1", "ck2", b"key")
    c = composite_bytes("m2", "ck1", b"key")
    d = composite_bytes("", "", b"key")
    assert a != b != c
    assert a != d


def test_composite_bytes_handles_collision_edge_case():
    """NUL separators prevent the boundary-collision where
    `model='abc'+'def'` and `model='ab'+'cdef'` would otherwise
    hash to the same string."""
    a = composite_bytes("abc", "def", b"k")
    b = composite_bytes("ab", "cdef", b"k")
    assert a != b


# ----------------------------------------------------------------------
# Single-key put/get round-trip
# ----------------------------------------------------------------------


def test_put_get_roundtrips_through_shard(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        payload = b"hello stripe" + b"\x00" * 100
        ok, reason = r.put(b"key-A", payload, retention="long", model="m", compat_key="ck")
        assert ok, reason
        got = r.get_bytes(b"key-A", model="m", compat_key="ck")
        assert got == payload
    finally:
        r.shutdown()


def test_put_get_works_regardless_of_picked_shard(tmp_path: Path):
    """Spray 16 keys (covers every shard with high probability for
    N=4); each must roundtrip through the same shard the hash picks."""
    r = _make_striped(tmp_path, n=4)
    try:
        for i in range(16):
            k = f"k{i:02d}".encode()
            v = f"v{i:02d}".encode() + b"x" * 32
            ok, reason = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok, (i, reason)
            assert r.get_bytes(k, model="m", compat_key="ck") == v
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Distribution — hash routing spreads keys across shards
# ----------------------------------------------------------------------


def test_keys_distributed_across_shards(tmp_path: Path):
    """Put 1000 keys, assert each shard got roughly 250 (each within
    30% of expected). blake2b is overkill cryptographically but means
    the distribution will be tight."""
    n = 4
    r = _make_striped(
        tmp_path, n=n, max_bytes=4 * 1024 * 1024, slot_bytes=4096, container_bytes=64 * 1024
    )
    try:
        for i in range(1000):
            k = f"key-{i:05d}".encode()
            ok, _ = r.put(k, b"v" * 64, retention="long", model="m", compat_key="ck")
            assert ok
        counts = [s.entries_count for s in r.shards]
        expected = 1000 / n
        for c in counts:
            # 30% bounds: blake2b at 1000 samples / 4 buckets should
            # comfortably stay within ±75 of 250.
            assert 0.7 * expected <= c <= 1.3 * expected, counts
    finally:
        r.shutdown()


def test_routing_is_deterministic_across_instances(tmp_path: Path):
    """Same composite must pick the same shard across two
    independently-constructed StripedLongRegion instances. The
    load-bearing property: blake2b is NOT salted (unlike CPython's
    built-in hash()). If this regresses, every cached block dies
    on kvd restart."""
    r1 = _make_striped(tmp_path / "a", n=8)
    r2 = _make_striped(tmp_path / "b", n=8)
    try:
        for i in range(50):
            k = f"k{i}".encode()
            c = composite_bytes("m", "ck", k)
            assert r1._pick(c) == r2._pick(c)
    finally:
        r1.shutdown()
        r2.shutdown()


# ----------------------------------------------------------------------
# Multi-key API
# ----------------------------------------------------------------------


def test_get_bytes_batch_returns_in_input_order(tmp_path: Path):
    """32 keys mixed across shards — output order must match input
    order regardless of which shard supplied which value (and the
    ThreadPoolExecutor's job-completion order)."""
    r = _make_striped(tmp_path, n=4)
    try:
        keys = []
        values = []
        for i in range(32):
            k = f"k-{i:02d}".encode()
            v = f"v-{i:02d}-padding".encode() + b"x" * 16
            keys.append(k)
            values.append(v)
            ok, _ = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok
        got = r.get_bytes_batch(keys, model="m", compat_key="ck")
        assert got == values
    finally:
        r.shutdown()


def test_get_bytes_batch_handles_misses(tmp_path: Path):
    """Mix of present + absent keys: misses must come back as None at
    the correct positions in the output list."""
    r = _make_striped(tmp_path, n=4)
    try:
        present_keys = [f"present-{i}".encode() for i in range(8)]
        present_values = [f"v{i}".encode() + b"x" * 32 for i in range(8)]
        for k, v in zip(present_keys, present_values, strict=True):
            ok, _ = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok
        # Interleave: [p0, miss, p1, miss, p2, miss, ...]
        absent_keys = [f"absent-{i}".encode() for i in range(8)]
        request = []
        expected = []
        for i in range(8):
            request.append(present_keys[i])
            expected.append(present_values[i])
            request.append(absent_keys[i])
            expected.append(None)
        got = r.get_bytes_batch(request, model="m", compat_key="ck")
        assert got == expected
    finally:
        r.shutdown()


def test_get_bytes_batch_empty_input(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        assert r.get_bytes_batch([], model="m", compat_key="ck") == []
    finally:
        r.shutdown()


def test_get_bytes_batch_uses_shard_batch_if_present(tmp_path: Path, monkeypatch):
    """When the underlying TablespaceLongRegion exposes a get_bytes_batch
    method, the striped region must call it preferentially rather than
    looping over single-key get_bytes."""
    r = _make_striped(tmp_path, n=2)
    try:
        ok, _ = r.put(b"k1", b"v1" + b"x" * 16, retention="long", model="m", compat_key="ck")
        assert ok
        ok, _ = r.put(b"k2", b"v2" + b"x" * 16, retention="long", model="m", compat_key="ck")
        assert ok

        # Monkey-patch one shard to expose a fake get_bytes_batch
        # that returns sentinels — we should see them in the output.
        called = {"n": 0}

        def fake_batch(keys, *, model, compat_key):
            called["n"] += 1
            return [b"SENTINEL"] * len(keys)

        for s in r.shards:
            s.get_bytes_batch = fake_batch  # type: ignore[attr-defined]

        got = r.get_bytes_batch([b"k1", b"k2"], model="m", compat_key="ck")
        assert all(v == b"SENTINEL" for v in got)
        assert called["n"] >= 1
    finally:
        r.shutdown()


def test_exists_returns_correct_per_key(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        for i in range(8):
            ok, _ = r.put(f"k{i}".encode(), b"v" * 32, retention="long", model="m", compat_key="ck")
            assert ok
        keys = [f"k{i}".encode() for i in range(8)] + [f"absent{i}".encode() for i in range(8)]
        result = r.exists(keys, model="m", compat_key="ck")
        assert result == [True] * 8 + [False] * 8
    finally:
        r.shutdown()


def test_exists_empty_input(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        assert r.exists([], model="m", compat_key="ck") == []
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Remove + clear
# ----------------------------------------------------------------------


def test_remove_works_through_correct_shard(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        ok, _ = r.put(b"k", b"v" * 64, retention="long", model="m", compat_key="ck")
        assert ok
        assert r.get_bytes(b"k", model="m", compat_key="ck") is not None
        assert r.remove(b"k", model="m", compat_key="ck") is True
        assert r.get_bytes(b"k", model="m", compat_key="ck") is None
        # Idempotent.
        assert r.remove(b"k", model="m", compat_key="ck") is False
    finally:
        r.shutdown()


def test_clear_clears_all_shards(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        for i in range(16):
            ok, _ = r.put(f"k{i}".encode(), b"v" * 32, retention="long", model="m", compat_key="ck")
            assert ok
        assert r.entries_count == 16
        dropped = r.clear()
        assert dropped == 16
        assert r.entries_count == 0
        for s in r.shards:
            assert s.entries_count == 0
    finally:
        r.shutdown()


def test_clear_namespace_only_clears_matching(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        for i in range(8):
            r.put(f"k{i}".encode(), b"v" * 32, retention="long", model="m1", compat_key="ck1")
            r.put(f"k{i}".encode(), b"v" * 32, retention="long", model="m2", compat_key="ck2")
        assert r.entries_count == 16
        dropped = r.clear_namespace("m1", "ck1")
        assert dropped == 8
        assert r.entries_count == 8
        # The m2/ck2 entries are still there.
        for i in range(8):
            assert r.get_bytes(f"k{i}".encode(), model="m2", compat_key="ck2") is not None
            assert r.get_bytes(f"k{i}".encode(), model="m1", compat_key="ck1") is None
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Stats aggregation
# ----------------------------------------------------------------------


def test_stats_aggregates_across_shards(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        for i in range(16):
            ok, _ = r.put(f"k{i}".encode(), b"v" * 64, retention="long", model="m", compat_key="ck")
            assert ok
        stats = r.stats()
        assert stats["num_shards"] == 4
        assert stats["entries_total"] == 16
        assert stats["bytes_used_total"] == sum(s.used_bytes for s in r.shards)
        assert stats["bytes_max_total"] == sum(s.max_bytes for s in r.shards)
        assert len(stats["shards"]) == 4
        # Each per-shard dict carries the expected keys.
        for sh in stats["shards"]:
            assert {
                "shard_id",
                "path",
                "max_bytes",
                "used_bytes",
                "entries_count",
                "slot_bytes",
                "num_containers",
            }.issubset(sh.keys())
    finally:
        r.shutdown()


def test_max_bytes_is_sum_across_shards(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        assert r.max_bytes == 4 * 128 * 1024
    finally:
        r.shutdown()


def test_used_bytes_is_sum_across_shards(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    try:
        ok, _ = r.put(b"k", b"x" * 100, retention="long", model="m", compat_key="ck")
        assert ok
        assert r.used_bytes == sum(s.used_bytes for s in r.shards) == 100
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_shutdown_shuts_all_shards_and_executor(tmp_path: Path):
    r = _make_striped(tmp_path, n=4)
    # Submit a stub job before shutdown to confirm the executor was
    # actually live during operation.
    fut = r._executor.submit(lambda: 42)
    assert fut.result() == 42

    r.shutdown()
    # All shards report not-started.
    for s in r.shards:
        assert s._started is False
    # Owned executor: a second submit should raise (executor shut
    # down) — we use a tiny lambda to provoke it.
    with pytest.raises(RuntimeError):
        r._executor.submit(lambda: None)


def test_start_is_idempotent(tmp_path: Path):
    r = _make_striped(tmp_path, n=2)
    try:
        # Already started by _make_striped; calling again is a no-op.
        r.start()
        ok, _ = r.put(b"k", b"v" * 32, retention="long", model="m", compat_key="ck")
        assert ok
    finally:
        r.shutdown()


def test_shutdown_is_idempotent(tmp_path: Path):
    r = _make_striped(tmp_path, n=2)
    r.shutdown()
    r.shutdown()  # No throw.


def test_external_executor_not_shut_down(tmp_path: Path):
    """If the caller passes an executor, StripedLongRegion must not
    shut it down on shutdown() — the caller owns its lifetime."""
    exec_ = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-extern")
    try:
        shards = [_make_shard(tmp_path / f"s{i}") for i in range(2)]
        r = StripedLongRegion(shards, executor=exec_)
        r.start()
        ok, _ = r.put(b"k", b"v" * 32, retention="long", model="m", compat_key="ck")
        assert ok
        r.shutdown()
        # Executor should still be alive — submit must succeed.
        fut = exec_.submit(lambda: "ok")
        assert fut.result() == "ok"
    finally:
        exec_.shutdown(wait=True)


# ----------------------------------------------------------------------
# Edge case: N = 1
# ----------------------------------------------------------------------


def test_single_shard_n1_works(tmp_path: Path):
    """N=1 degenerates to a thin wrapper around one TablespaceLongRegion.
    No fanout, but should still function — the fanout-overhead delta
    is what we'd be measuring as 'cost of striping when you don't need
    it' in the bench."""
    r = _make_striped(tmp_path, n=1)
    try:
        for i in range(8):
            ok, _ = r.put(f"k{i}".encode(), b"v" * 32, retention="long", model="m", compat_key="ck")
            assert ok
        got = r.get_bytes_batch([f"k{i}".encode() for i in range(8)], model="m", compat_key="ck")
        assert all(v is not None for v in got)
        assert r.entries_count == 8
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Intra-shard worker pool (workers_per_shard fanout)
# ----------------------------------------------------------------------


def test_workers_per_shard_dispatches_sub_batches(tmp_path: Path):
    """16 keys all routed to one shard, workers_per_shard=4: each
    per-shard get takes 10 ms (mocked). Serial wall would be 160 ms;
    4-way fanout wall should be ~40 ms baseline + scheduler slack.

    We assert wall < 100 ms (well below 160 ms serial) and a strict
    minimum of 4 distinct executor threads observed in the get_bytes
    call site — proving that the sub-pool actually fanned out rather
    than serializing on one worker.
    """
    import threading
    import time

    # 1 shard so every key routes to the same place — that's the
    # condition where intra-shard parallelism matters most.
    r = _make_striped(tmp_path, n=1)
    try:
        # Override workers_per_shard explicitly; the default is 4 but
        # we want the test to fail loudly if that default changes.
        r._workers_per_shard = 4

        # Populate 16 keys.
        n_keys = 16
        for i in range(n_keys):
            ok, _ = r.put(
                f"k{i:02d}".encode(),
                b"v" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok

        # Monkey-patch the single shard's get_bytes to sleep 10 ms.
        # We track which threads invoke it — with 4 sub-workers we
        # expect at least 4 distinct kvd-stripe threads.
        per_call_delay_s = 0.010
        seen_threads: set[str] = set()
        seen_lock = threading.Lock()
        real_get = r._shards[0].get_bytes

        def slow_get(key, *, model="", compat_key=""):
            with seen_lock:
                seen_threads.add(threading.current_thread().name)
            time.sleep(per_call_delay_s)
            return real_get(key, model=model, compat_key=compat_key)

        r._shards[0].get_bytes = slow_get  # type: ignore[method-assign]
        # Remove any native batched read so _shard_batch_get falls back
        # to looping get_bytes (otherwise the patched per-call sleep
        # wouldn't fire for every key).
        if hasattr(r._shards[0], "get_bytes_batch"):
            object.__setattr__(r._shards[0], "get_bytes_batch", None)

        keys = [f"k{i:02d}".encode() for i in range(n_keys)]
        t0 = time.perf_counter()
        got = r.get_bytes_batch(keys, model="m", compat_key="ck")
        elapsed = time.perf_counter() - t0

        # Correctness: every key returned a value.
        assert all(v is not None for v in got)
        assert len(got) == n_keys

        # Parallelism: wall must be substantially below the 160 ms
        # serial bound. The 4-way ideal is 40 ms + scheduler slack;
        # 100 ms gives a 2.5× safety margin for CI variance while
        # still flunking any version that runs serially.
        assert elapsed < 0.100, (
            f"4 sub-workers should finish 16 × 10 ms in <100 ms, "
            f"got {elapsed:.3f}s — sub-batches may be serializing"
        )
        # And it must be at least the single-call latency (sanity:
        # the sleep actually ran).
        assert elapsed >= per_call_delay_s, (
            f"wall {elapsed:.3f}s below single-call {per_call_delay_s}s — "
            f"the slow-get shim didn't fire"
        )

        # 4 distinct executor threads must have served the load.
        assert len(seen_threads) >= 4, (
            f"expected >=4 distinct executor threads, got {len(seen_threads)}: {seen_threads}"
        )
    finally:
        r.shutdown()


def test_workers_per_shard_one_falls_back_to_single_future(tmp_path: Path):
    """workers_per_shard=1 reproduces the pre-sub-pool behavior:
    exactly one future per non-empty shard. We verify by counting
    executor submissions via a wrapping shim."""
    shards = [_make_shard(tmp_path / f"s{i}") for i in range(4)]
    r = StripedLongRegion(shards, workers_per_shard=1)
    r.start()
    try:
        # Spread 16 keys across the 4 shards (~4 keys/shard on
        # average with blake2b distribution).
        keys = []
        for i in range(16):
            k = f"k{i:02d}".encode()
            keys.append(k)
            ok, _ = r.put(k, b"v" * 64, retention="long", model="m", compat_key="ck")
            assert ok

        # Count futures by wrapping submit. We intercept on the
        # executor itself — only get_bytes_batch's submits count
        # because that's the only call site here.
        real_submit = r._executor.submit
        submit_count = {"n": 0}

        def counting_submit(fn, *a, **kw):
            submit_count["n"] += 1
            return real_submit(fn, *a, **kw)

        r._executor.submit = counting_submit  # type: ignore[method-assign]
        got = r.get_bytes_batch(keys, model="m", compat_key="ck")

        # Correctness.
        assert len(got) == len(keys)
        assert all(v is not None for v in got)

        # Exactly one future per non-empty shard. With 16 keys
        # blake2b-routed to 4 shards, every shard gets at least one
        # key — so submit_count must equal 4.
        non_empty_shards = sum(1 for s in r.shards if s.entries_count > 0)
        assert submit_count["n"] == non_empty_shards, (
            f"workers_per_shard=1 should emit one future per non-empty "
            f"shard ({non_empty_shards}), got {submit_count['n']}"
        )
    finally:
        r.shutdown()


def test_workers_per_shard_handles_uneven_chunks(tmp_path: Path):
    """5 keys to one shard with workers_per_shard=4: ceil(5/4)=2,
    so the split is 2+2+1+0 → the empty chunk is skipped and all
    5 keys still come back in input order."""
    r = _make_striped(tmp_path, n=1)
    try:
        r._workers_per_shard = 4
        keys = []
        values = []
        for i in range(5):
            k = f"k{i}".encode()
            v = f"v{i}".encode() + b"x" * 32
            keys.append(k)
            values.append(v)
            ok, _ = r.put(k, v, retention="long", model="m", compat_key="ck")
            assert ok

        got = r.get_bytes_batch(keys, model="m", compat_key="ck")
        # Order preservation: i-th input must yield i-th value
        # regardless of which sub-worker handled it.
        assert got == values
    finally:
        r.shutdown()


def test_workers_per_shard_handles_more_workers_than_keys(tmp_path: Path):
    """2 keys to one shard with workers_per_shard=8: ceil(2/8)=1,
    we should submit at most 2 futures (one per key), not 8 empty
    ones."""
    shards = [_make_shard(tmp_path / "s0")]
    r = StripedLongRegion(shards, workers_per_shard=8)
    r.start()
    try:
        for i in range(2):
            ok, _ = r.put(
                f"k{i}".encode(),
                b"v" * 32,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok

        real_submit = r._executor.submit
        count = {"n": 0}

        def counting_submit(fn, *a, **kw):
            count["n"] += 1
            return real_submit(fn, *a, **kw)

        r._executor.submit = counting_submit  # type: ignore[method-assign]
        got = r.get_bytes_batch([b"k0", b"k1"], model="m", compat_key="ck")
        assert all(v is not None for v in got)
        # At most 2 futures (one per non-empty chunk), strictly less
        # than the worker pool size of 8.
        assert count["n"] <= 2, (
            f"workers_per_shard=8 with 2 keys should emit <=2 futures, got {count['n']}"
        )
    finally:
        r.shutdown()


def test_executor_size_scales_with_workers_per_shard(tmp_path: Path):
    """Auto-created executor must have n_shards × workers_per_shard
    threads, otherwise we'd silently throttle the fanout at the
    executor level."""
    shards = [_make_shard(tmp_path / f"s{i}") for i in range(3)]
    r = StripedLongRegion(shards, workers_per_shard=5)
    try:
        assert r._executor._max_workers == 3 * 5
        assert r.workers_per_shard == 5
    finally:
        r.shutdown()


def test_workers_per_shard_rejects_zero(tmp_path: Path):
    shards = [_make_shard(tmp_path / "s0")]
    with pytest.raises(ValueError, match="workers_per_shard"):
        StripedLongRegion(shards, workers_per_shard=0)


def test_workers_per_shard_default_is_eight(tmp_path: Path):
    """Default contract: workers_per_shard=8 unless overridden. This
    is the empirical knee of the throughput curve on 8-NVMe (see
    bench): 1→4 climbs slowly, 4→8 jumps from 6.3→9.6 GB/s, 8→16 is
    marginal. Guards against accidental default drift between
    releases."""
    shards = [_make_shard(tmp_path / "s0")]
    r = StripedLongRegion(shards)
    try:
        assert r.workers_per_shard == 8
    finally:
        r.shutdown()


# ----------------------------------------------------------------------
# Restart-survival via deterministic hash
# ----------------------------------------------------------------------


def test_restart_preserves_routing(tmp_path: Path):
    """Critical: after kvd restart, every key must hash to the SAME
    shard or its tablespace journal is in the wrong shard. This is the
    reason we use hashlib.blake2b instead of Python's salted hash()."""
    # Phase 1: write a known set of keys.
    shards_a = [_make_shard(tmp_path / f"s{i}") for i in range(4)]
    r_a = StripedLongRegion(shards_a)
    r_a.start()
    try:
        for i in range(32):
            ok, _ = r_a.put(
                f"k{i:02d}".encode(),
                f"v{i:02d}".encode() + b"x" * 64,
                retention="long",
                model="m",
                compat_key="ck",
            )
            assert ok
    finally:
        r_a.shutdown()

    # Phase 2: re-open and read back. The same hash MUST resolve to
    # the same shard so the journal-replayed entry is findable.
    shards_b = [_make_shard(tmp_path / f"s{i}") for i in range(4)]
    r_b = StripedLongRegion(shards_b)
    r_b.start()
    try:
        for i in range(32):
            got = r_b.get_bytes(f"k{i:02d}".encode(), model="m", compat_key="ck")
            assert got == f"v{i:02d}".encode() + b"x" * 64
    finally:
        r_b.shutdown()
