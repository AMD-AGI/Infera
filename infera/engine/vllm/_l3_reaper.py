###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""L3 file-tier reaper for the kvd connector.

Issue #55: the connector-owned hipFile L3 tier (chunk files under
``hipfile_roots``) had no enforcement of total file volume.
``--long-bytes`` on the daemon never applied to connector-written files,
so a long sweep could grow the L3 dir past the configured budget and
fill the underlying filesystem — observed on a shared NFS where one run
grew the L3 dir to 4.8 TB before pushing the FS to 100 % full and
collapsing throughput from 152k to 30k tok/s.

This module owns the file-volume accounting and eviction. The connector
calls ``register(path, size, retention)`` after every successful save,
``touch(path)`` on every successful load (for LRU-on-read), and
``unregister(path)`` if it ever explicitly unlinks a file. A daemon
thread ticks every ``interval_s`` seconds: ``statvfs`` each root; if
``used > budget`` OR ``free < floor`` for the root's FS, evict by
(retention-priority, LRU-mtime) until both bounds are satisfied.

Free-space-keyed eviction matters more than budget-keyed on a shared FS
where a neighbor can fill the disk even while we stay under our budget.
Three layers:

  * Periodic reaper (this thread)            — primary
  * ENOSPC backstop in the save path          — burst-recovery
  * Startup free-space detect + budget clamp  — config sanity

This module intentionally does not import torch / hipfile — pure stdlib
so import cost is trivial and the reaper survives partial-install
environments (e.g. unit tests with no GPU).
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger("infera.engine.vllm.l3_reaper")


# Retention-priority order. Lower index = LOWER priority = evict first.
# ``short`` is the spillover tier (intended to churn), so it gets evicted
# ahead of ``long`` whenever budget pressure forces a choice.
_RETENTION_EVICT_ORDER = ("short", "long")


def _retention_rank(retention: str) -> int:
    try:
        return _RETENTION_EVICT_ORDER.index(retention)
    except ValueError:
        # Unknown retention -> highest priority (last to evict). The
        # connector never invents new retention strings, but be safe.
        return len(_RETENTION_EVICT_ORDER)


class L3FileReaper:
    """Background-thread file-tier reaper.

    Thread-safe. All public methods acquire ``self._lock`` so concurrent
    save/load workers can call without external synchronization.

    Args:
        roots:           {retention: dir_path} — same dict the connector
                         was constructed with.
        budget_bytes:    total file-volume budget summed across roots.
                         ``0`` or negative disables budget-keyed eviction
                         (free-space-keyed eviction still runs).
        free_floor_ratio:fraction of each root FS that must stay free.
                         Reaper evicts until ``free >= total*ratio``.
                         Default 0.05 (5 %).
        interval_s:      reaper tick period. Default 30 s. Set <=0 to
                         disable the background thread (callers can still
                         drive eviction manually via ``reap_once()``).
        clock:           injectable for tests (returns monotonic-ish
                         seconds). Default ``time.time``.
        statvfs:         injectable for tests; same signature as
                         ``os.statvfs``. Default ``os.statvfs``.
        unlink:          injectable for tests; same signature as
                         ``os.unlink``. Default ``os.unlink``.
    """

    def __init__(
        self,
        roots: dict[str, str],
        budget_bytes: int,
        *,
        free_floor_ratio: float = 0.05,
        interval_s: float = 30.0,
        clock=None,
        statvfs=None,
        unlink=None,
    ) -> None:
        # roots[retention] → directory (one or more retentions can share
        # a path; we dedupe by path for FS-level checks but keep the
        # retention label per entry for eviction priority).
        self._roots = dict(roots)
        self._budget = int(budget_bytes) if budget_bytes else 0
        self._floor_ratio = max(0.0, min(1.0, float(free_floor_ratio)))
        self._interval = float(interval_s)
        self._now = clock or time.time
        self._statvfs = statvfs or os.statvfs
        self._unlink = unlink or os.unlink

        # path -> (size_bytes, mtime_seconds, retention)
        self._entries: dict[str, tuple[int, float, str]] = {}
        self._used_bytes = 0
        # Per-root tracking — sum of sizes for files under that root.
        # Used so free-space-driven eviction can prefer files on the
        # offending FS rather than the global LRU.
        self._used_by_root: dict[str, int] = {r: 0 for r in self._roots.values()}

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Stats (read-only externally; protected by self._lock).
        self.stats = {
            "reap_ticks": 0,
            "evictions": 0,
            "bytes_evicted": 0,
            "enospc_recoveries": 0,
            "register_skipped_nonexistent": 0,
        }

    # --------------------------------------------------------------- registry

    def register(self, path: str, size_bytes: int, retention: str = "long") -> None:
        """Add a freshly-published chunk file to the registry.

        Idempotent: re-registering an existing path replaces the old
        entry (the connector currently never rewrites a content-hashed
        path, but be safe). ``size_bytes`` must be the on-disk file size
        (header + payload), not just the payload.
        """
        if size_bytes <= 0:
            return
        with self._lock:
            prev = self._entries.get(path)
            if prev is not None:
                self._used_bytes -= prev[0]
                root = self._root_for_path(path)
                if root is not None:
                    self._used_by_root[root] = max(0, self._used_by_root[root] - prev[0])
            self._entries[path] = (int(size_bytes), self._now(), retention)
            self._used_bytes += size_bytes
            root = self._root_for_path(path)
            if root is not None:
                self._used_by_root[root] = self._used_by_root.get(root, 0) + size_bytes

    def unregister(self, path: str) -> None:
        """Drop a path from the registry (called when the connector
        unlinks a file explicitly — e.g. failed-tmp cleanup). Safe to
        call on an unknown path."""
        with self._lock:
            prev = self._entries.pop(path, None)
            if prev is None:
                return
            self._used_bytes = max(0, self._used_bytes - prev[0])
            root = self._root_for_path(path)
            if root is not None:
                self._used_by_root[root] = max(0, self._used_by_root[root] - prev[0])

    def touch(self, path: str) -> None:
        """Bump the entry's mtime to ``now`` (LRU-on-read). Safe on
        unknown paths."""
        with self._lock:
            prev = self._entries.get(path)
            if prev is None:
                return
            size, _, retention = prev
            self._entries[path] = (size, self._now(), retention)

    def scan_existing(self, *, suffix: str = ".kvcache") -> int:
        """One-time startup scan: walk each root, register every existing
        ``*.kvcache`` file. Returns the count registered. Files seen here
        get the file's actual ``st_mtime`` so warm chunks from a prior
        run keep their natural LRU age."""
        count = 0
        seen_roots: set[str] = set()
        for retention, root in self._roots.items():
            if not root or root in seen_roots:
                continue
            seen_roots.add(root)
            try:
                for dirpath, _, filenames in os.walk(root):
                    for fn in filenames:
                        if not fn.endswith(suffix):
                            continue
                        p = os.path.join(dirpath, fn)
                        try:
                            stat = os.stat(p)
                        except OSError:
                            continue
                        with self._lock:
                            self._entries[p] = (stat.st_size, stat.st_mtime, retention)
                            self._used_bytes += stat.st_size
                            self._used_by_root[root] = (
                                self._used_by_root.get(root, 0) + stat.st_size
                            )
                        count += 1
            except OSError as exc:
                logger.warning("l3 reaper: scan_existing failed for root %s: %s", root, exc)
        return count

    # ----------------------------------------------------------- thread mgmt

    def start(self) -> None:
        if self._thread is not None or self._interval <= 0:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="kvd-l3-reaper",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.reap_once()
            except Exception:
                logger.exception("l3 reaper: tick failed")

    # ---------------------------------------------------------------- reap

    def reap_once(self, *, force_evict_bytes: int | None = None) -> int:
        """One reap pass. Returns total bytes evicted.

        Evicts when ANY of:
          * ``self._used_bytes > self._budget`` (when budget enabled)
          * any root's FS has ``free < total * floor_ratio``
          * ``force_evict_bytes`` is set (used by the ENOSPC backstop —
            forces at least that many bytes free even if the FS appears
            to have room, because the actual write that hit ENOSPC may
            be larger than reported free due to allocator overhead).
        """
        evicted_total = 0
        with self._lock:
            self.stats["reap_ticks"] += 1

        # 1. Free-space pressure per root.
        seen_roots: set[str] = set()
        for root in self._roots.values():
            if not root or root in seen_roots:
                continue
            seen_roots.add(root)
            need_free = self._free_shortfall(root)
            if need_free > 0:
                evicted_total += self._evict_for_root(root, need_free)

        # 2. Global budget pressure.
        if self._budget > 0 and self._used_bytes > self._budget:
            target = int(self._budget * 0.9)  # evict 10 % below budget for hysteresis
            evicted_total += self._evict_global_to(target)

        # 3. Force eviction (ENOSPC backstop).
        if force_evict_bytes is not None and force_evict_bytes > 0:
            evicted_total += self._evict_global_to(max(0, self._used_bytes - force_evict_bytes))

        return evicted_total

    def _free_shortfall(self, root: str) -> int:
        """Bytes we need to free under ``root`` to bring its FS up to
        the floor. 0 if already at/above floor or the floor is disabled.
        Returns 0 on statvfs error (best-effort)."""
        if self._floor_ratio <= 0:
            return 0
        try:
            stat = self._statvfs(root)
        except OSError:
            return 0
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        floor = int(total * self._floor_ratio)
        return max(0, floor - free)

    def _evict_for_root(self, root: str, bytes_needed: int) -> int:
        """Evict LRU+priority entries WHOSE PATH STARTS WITH ``root``
        until ``bytes_needed`` bytes have been freed. Prefers ``short``
        retention then by mtime (oldest first). Returns bytes freed."""
        freed = 0
        with self._lock:
            candidates = sorted(
                (
                    (p, sz, mt, ret)
                    for p, (sz, mt, ret) in self._entries.items()
                    if self._path_under_root(p, root)
                ),
                key=lambda x: (_retention_rank(x[3]), x[2]),
            )
            for path, size, _mt, _retention in candidates:
                if freed >= bytes_needed:
                    break
                if self._evict_one_locked(path):
                    freed += size
        return freed

    def _evict_global_to(self, target_used_bytes: int) -> int:
        """Evict until ``self._used_bytes <= target_used_bytes``,
        regardless of root. Same (retention, LRU) order. Returns bytes
        freed."""
        freed = 0
        with self._lock:
            if self._used_bytes <= target_used_bytes:
                return 0
            candidates = sorted(
                self._entries.items(),
                key=lambda kv: (_retention_rank(kv[1][2]), kv[1][1]),
            )
            for path, (size, _mt, _ret) in candidates:
                if self._used_bytes <= target_used_bytes:
                    break
                if self._evict_one_locked(path):
                    freed += size
        return freed

    def _evict_one_locked(self, path: str) -> bool:
        """Unlink the file + drop the registry entry. Caller holds the
        lock. Returns True if evicted (file unlinked OR already gone)."""
        entry = self._entries.get(path)
        if entry is None:
            return False
        size, _mt, _ret = entry
        try:
            self._unlink(path)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                logger.warning("l3 reaper: unlink %s failed: %s", path, exc)
                # Leave the entry in place; we'll retry next tick. Don't
                # decrement used_bytes since the file is still on disk.
                return False
            # Already gone — fall through and drop the entry.
        # Drop registry entry + accounting.
        del self._entries[path]
        self._used_bytes = max(0, self._used_bytes - size)
        root = self._root_for_path(path)
        if root is not None:
            self._used_by_root[root] = max(0, self._used_by_root[root] - size)
        self.stats["evictions"] += 1
        self.stats["bytes_evicted"] += size
        return True

    # ---------------------------------------------------------- ENOSPC hook

    def on_enospc(self, *, want_free_bytes: int) -> int:
        """ENOSPC backstop: a save just hit ENOSPC. Run an emergency
        reap aiming to free ``want_free_bytes`` and return what was
        actually freed. The caller should retry the write iff
        ``returned >= want_free_bytes`` (else give up gracefully and
        let the prefix be re-prefilled by the engine)."""
        with self._lock:
            self.stats["enospc_recoveries"] += 1
        return self.reap_once(force_evict_bytes=want_free_bytes)

    # ---------------------------------------------------------------- helpers

    def _root_for_path(self, path: str) -> str | None:
        """Return the configured root directory that ``path`` lives
        under, or None if none match. (One pass over roots; the count
        is tiny — usually 1 or 2.)"""
        for r in self._roots.values():
            if r and self._path_under_root(path, r):
                return r
        return None

    @staticmethod
    def _path_under_root(path: str, root: str) -> bool:
        # Normalize trailing slashes so a path equal to the root + "/x"
        # is recognized regardless of whether root ended with "/".
        r = root.rstrip("/") + "/"
        return path.startswith(r)

    # ---------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, object]:
        """Read-only snapshot of accounting state. Useful for tests +
        ``/metrics``-style exposure."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "used_bytes": self._used_bytes,
                "budget_bytes": self._budget,
                "free_floor_ratio": self._floor_ratio,
                "interval_s": self._interval,
                "used_by_root": dict(self._used_by_root),
                "stats": dict(self.stats),
            }


def startup_budget_clamp(
    roots: Iterable[str],
    declared_budget_bytes: int,
    *,
    free_floor_ratio: float = 0.05,
    statvfs=None,
) -> tuple[int, list[str]]:
    """One-shot startup helper: walk each root, ``statvfs`` it, and
    return the *effective* budget = ``min(declared, smallest_free * 0.9)``
    plus a list of warning strings (one per root that's already below the
    free-space floor — the caller should log them and decide whether to
    proceed).

    Pure function — does not mutate FS or start the reaper; the connector
    builds an ``L3FileReaper`` with the effective budget afterwards.
    """
    sv = statvfs or os.statvfs
    warnings: list[str] = []
    smallest_free = None
    seen: set[str] = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        try:
            stat = sv(root)
        except OSError as exc:
            warnings.append(f"statvfs({root}) failed: {exc}")
            continue
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        if total > 0 and free < int(total * free_floor_ratio):
            warnings.append(
                f"root {root}: free {free / 1e9:.2f} GB < floor "
                f"{free_floor_ratio * 100:.1f} % of {total / 1e9:.2f} GB"
            )
        if smallest_free is None or free < smallest_free:
            smallest_free = free
    effective = declared_budget_bytes
    if smallest_free is not None:
        cap = int(smallest_free * 0.9)
        if declared_budget_bytes <= 0 or cap < declared_budget_bytes:
            effective = cap
    return effective, warnings
