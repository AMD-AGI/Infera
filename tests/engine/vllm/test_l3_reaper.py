###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the L3 file-tier reaper (issue #55).

Pure-stdlib reaper -> no GPU / no hipfile / no torch needed. Uses
``tmp_path`` fixtures + an injectable clock so the LRU-mtime ordering
is deterministic.
"""

from __future__ import annotations

import errno
import os
from typing import Any

from infera.engine.vllm._l3_reaper import (
    L3FileReaper,
    startup_budget_clamp,
)


class _FakeClock:
    """Monotonic-ish injectable clock."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += float(dt)


class _FakeStatvfs:
    """Injectable statvfs that lies about (total, free) per path."""

    def __init__(self, per_path: dict[str, tuple[int, int]]) -> None:
        # per_path[path] = (f_blocks, f_bavail) in 4 KiB blocks for simplicity
        self._per = per_path

    def __call__(self, path: str) -> Any:
        if path not in self._per:
            raise OSError(errno.ENOENT, "no such root")
        f_blocks, f_bavail = self._per[path]

        class _R:
            pass

        r = _R()
        r.f_blocks = f_blocks
        r.f_bavail = f_bavail
        r.f_frsize = 4096
        return r


def _touch(path: str, size: int = 4096) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


# ----------------------------------------------------------- register / unregister


def test_register_and_unregister_track_used_bytes(tmp_path):
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(roots, budget_bytes=0, interval_s=0)
    p = str(tmp_path / "a.kvcache")
    _touch(p, 1024)
    r.register(p, 1024, retention="long")
    snap = r.snapshot()
    assert snap["entries"] == 1
    assert snap["used_bytes"] == 1024
    r.unregister(p)
    snap = r.snapshot()
    assert snap["entries"] == 0
    assert snap["used_bytes"] == 0


def test_register_idempotent_replaces_size(tmp_path):
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(roots, budget_bytes=0, interval_s=0)
    p = str(tmp_path / "a.kvcache")
    r.register(p, 1024, retention="long")
    r.register(p, 4096, retention="long")  # same path, new size
    assert r.snapshot()["used_bytes"] == 4096
    assert r.snapshot()["entries"] == 1


# --------------------------------------------------------------- scan_existing


def test_scan_existing_picks_up_files_left_over_from_prior_run(tmp_path):
    roots = {"long": str(tmp_path)}
    _touch(str(tmp_path / "a/b/x.kvcache"), 2048)
    _touch(str(tmp_path / "c/d/y.kvcache"), 1024)
    _touch(str(tmp_path / "junk.txt"), 999)  # ignored — wrong suffix
    r = L3FileReaper(roots, budget_bytes=0, interval_s=0)
    n = r.scan_existing()
    assert n == 2
    assert r.snapshot()["used_bytes"] == 2048 + 1024


# ------------------------------------------------------- budget-keyed eviction


def test_reap_evicts_oldest_when_over_budget(tmp_path):
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=10_000,
        interval_s=0,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (10_000, 9_000)}),
    )
    # Register five 3 KiB entries; budget 10K → over by 5K → evict oldest 2.
    for i in range(5):
        p = str(tmp_path / f"file_{i}.kvcache")
        _touch(p, 3000)
        r.register(p, 3000, retention="long")
        clock.advance(1.0)  # ensure distinct mtimes
    freed = r.reap_once()
    assert freed >= 5000
    # Oldest survives only if it's beyond the kept set.
    remaining = sorted(os.listdir(tmp_path))
    # We expect file_3 and file_4 (the two newest) to survive at minimum,
    # and at most one more (file_2) since budget*0.9 == 9000 = 3 entries.
    assert "file_4.kvcache" in remaining
    assert "file_3.kvcache" in remaining
    assert "file_0.kvcache" not in remaining
    assert "file_1.kvcache" not in remaining


def test_reap_prefers_short_retention_over_long(tmp_path):
    """``short`` is the spillover tier — evict it ahead of ``long`` even
    if ``long`` is older."""
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=5_000,
        interval_s=0,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (10_000, 9_000)}),
    )
    # OLDER long file first, then a NEWER short file.
    p_long = str(tmp_path / "old_long.kvcache")
    _touch(p_long, 3000)
    r.register(p_long, 3000, retention="long")
    clock.advance(100.0)
    p_short = str(tmp_path / "new_short.kvcache")
    _touch(p_short, 3000)
    r.register(p_short, 3000, retention="short")
    # Budget 5K, used 6K → must evict 1. Expect short to go despite being
    # newer.
    r.reap_once()
    assert os.path.exists(p_long)
    assert not os.path.exists(p_short)


# ---------------------------------------------------- free-space-keyed eviction


def test_reap_evicts_when_root_fs_below_floor(tmp_path):
    """Budget OK but the FS underneath the root is 99 % full → evict
    anyway so a neighbor doesn't fill the disk."""
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    # FS: 1 M blocks total, 1 K blocks free → 0.1 % free, floor 5 %.
    r = L3FileReaper(
        roots,
        budget_bytes=10**12,
        interval_s=0,
        free_floor_ratio=0.05,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (1_000_000, 1_000)}),
    )
    for i in range(3):
        p = str(tmp_path / f"file_{i}.kvcache")
        _touch(p, 4096)
        r.register(p, 4096, retention="long")
        clock.advance(1.0)
    freed = r.reap_once()
    # Need (1_000_000 - 1_000) * 4096 * 0.05 ≈ 204.6 GB worth of bytes
    # to satisfy the floor — we only have 12 KiB total → reaper evicts
    # everything it can.
    assert freed == 4096 * 3
    snap = r.snapshot()
    assert snap["entries"] == 0


def test_reap_no_evictions_when_under_budget_and_above_floor(tmp_path):
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=10_000,
        interval_s=0,
        free_floor_ratio=0.05,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (100, 80)}),  # 80 % free
    )
    p = str(tmp_path / "a.kvcache")
    _touch(p, 1024)
    r.register(p, 1024, retention="long")
    freed = r.reap_once()
    assert freed == 0
    assert os.path.exists(p)


# ------------------------------------------------------------ touch / LRU


def test_touch_promotes_entry_so_it_survives_next_eviction(tmp_path):
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=2_500,
        interval_s=0,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (10_000, 9_000)}),
    )
    paths = []
    for i in range(3):
        p = str(tmp_path / f"file_{i}.kvcache")
        _touch(p, 1000)
        r.register(p, 1000, retention="long")
        clock.advance(1.0)
        paths.append(p)
    # Touch the OLDEST to make it the newest.
    clock.advance(100.0)
    r.touch(paths[0])
    r.reap_once()  # budget 2500*0.9 = 2250 → keep 2 entries
    # paths[0] was touched → newest → must survive. paths[1] is now oldest.
    assert os.path.exists(paths[0])
    assert not os.path.exists(paths[1])


# ------------------------------------------------------------ ENOSPC backstop


def test_on_enospc_forces_eviction(tmp_path):
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=10**12,
        interval_s=0,  # huge budget — no budget pressure
        free_floor_ratio=0.0,  # no floor pressure
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (10_000, 9_000)}),
    )
    for i in range(4):
        p = str(tmp_path / f"file_{i}.kvcache")
        _touch(p, 1000)
        r.register(p, 1000, retention="long")
        clock.advance(1.0)
    freed = r.on_enospc(want_free_bytes=2500)
    assert freed >= 2500
    assert r.stats["enospc_recoveries"] == 1


# ---------------------------------------------------- unlink-already-gone


def test_eviction_handles_file_already_deleted(tmp_path):
    """A neighbor may have removed the file out from under us. Reaper
    should still drop the registry entry + accounting."""
    clock = _FakeClock()
    roots = {"long": str(tmp_path)}
    r = L3FileReaper(
        roots,
        budget_bytes=1000,
        interval_s=0,
        clock=clock,
        statvfs=_FakeStatvfs({str(tmp_path): (10_000, 9_000)}),
    )
    p = str(tmp_path / "ghost.kvcache")
    _touch(p, 4096)
    r.register(p, 4096, retention="long")
    os.unlink(p)  # external removal
    freed = r.reap_once()
    assert freed == 4096
    assert r.snapshot()["entries"] == 0
    assert r.snapshot()["used_bytes"] == 0


# ----------------------------------------------------- thread lifecycle


def test_start_stop_thread_no_op_when_interval_zero(tmp_path):
    r = L3FileReaper({"long": str(tmp_path)}, budget_bytes=0, interval_s=0)
    r.start()  # no-op (interval=0)
    assert r._thread is None
    r.stop()


def test_start_stop_thread_runs_and_terminates(tmp_path):
    r = L3FileReaper({"long": str(tmp_path)}, budget_bytes=0, interval_s=0.01)
    r.start()
    assert r._thread is not None
    r.stop(timeout=2.0)
    # After stop, thread joined and the slot is cleared.
    assert r._thread is None


# --------------------------------------------------------- startup clamp


def test_startup_budget_clamp_uses_smallest_free(tmp_path):
    roots = [str(tmp_path / "a"), str(tmp_path / "b")]
    for r in roots:
        os.makedirs(r)
    sv = _FakeStatvfs(
        {
            roots[0]: (10_000, 8_000),  # 8 K * 4096 = 32 MiB free
            roots[1]: (10_000, 4_000),  # 4 K * 4096 = 16 MiB free → smaller
        }
    )
    effective, warnings = startup_budget_clamp(
        roots,
        declared_budget_bytes=10**12,
        free_floor_ratio=0.05,
        statvfs=sv,
    )
    assert effective == int(4_000 * 4096 * 0.9)
    assert warnings == []


def test_startup_budget_clamp_warns_when_below_floor(tmp_path):
    roots = [str(tmp_path / "a")]
    os.makedirs(roots[0])
    sv = _FakeStatvfs({roots[0]: (10_000, 100)})  # 1 % free, floor 5 %
    effective, warnings = startup_budget_clamp(
        roots,
        declared_budget_bytes=10**9,
        free_floor_ratio=0.05,
        statvfs=sv,
    )
    assert len(warnings) == 1
    assert "below" not in warnings[0]  # we say "<" not "below"
    assert "<" in warnings[0]
    assert effective == int(100 * 4096 * 0.9)
