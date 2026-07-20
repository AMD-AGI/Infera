###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""In-memory KV store for infera-kvd Phase 3.0.

Single tier (host RAM). Capacity bound is `max_bytes`. Eviction
policy: **priority-aware LRU** — lowest retention level evicted
first, then within the same retention level, least-recently-accessed.

This mirrors SGLang's `PriorityStrategy` (we discovered it in code:
`python/sglang/srt/mem_cache/evict_policy.py:PriorityStrategy`).
Same shape, our keys are content hashes instead of TreeNodes.

What's intentionally NOT here (Phase 3.5+ work):
- SSD layer (spillover + long regions)
- Persistence / restart recovery
- Multi-tenant quotas
- Mooncake / NIXL transports

State machine on a SET:

  1. If key already present → update value, refresh last_access, ack.
  2. Else if free space available → insert, ack.
  3. Else find the lowest-priority victim with `last_access` ≤ incoming.
     If victim's priority is strictly lower → evict, insert, ack.
     If victim's priority is equal AND older → evict, insert, ack.
     Otherwise → reject (don't displace a higher-priority block).

This means a heavily-loaded daemon serving long-retention blocks
won't be displaced by a flood of short-retention writes. We document
the rejection case clearly so the engine can react (e.g. fall back
to direct GPU recomputation rather than waste the network round-trip).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

# Forward type-only import to keep store.py importable without ssd.py
# loaded eagerly (avoids circular-init issues).
from typing import TYPE_CHECKING

from infera.kvd.wire import (
    RETENTION_EPHEMERAL,
    RETENTION_LONG,
    RETENTION_NONE,
    RETENTION_SHORT,
    validate_retention,
)

if TYPE_CHECKING:
    from infera.kvd.shared_arena import SharedArena
    from infera.kvd.ssd import LongStorageRegion, SpilloverRegion
    from infera.kvd.tablespace import TablespaceLongRegion
    from infera.kvd.tablespace_multipool import (
        MultiPoolTablespaceLongRegion,  # noqa: F401 — used in the string-form type alias below
    )

# Type alias: the long-retention region can be any of three impls.
# All three expose the same public surface: start, shutdown,
# get_bytes, exists, get_entry, put, remove, clear, used_bytes,
# entries_count, max_bytes. HostStore treats them interchangeably.
#   - LongStorageRegion (Phase 4.0): file-per-block, simplest
#   - TablespaceLongRegion (Phase B): single-pool chunk allocator
#   - MultiPoolTablespaceLongRegion (Phase B.2+): N pools, smart routing
LongRegionLike = "LongStorageRegion | TablespaceLongRegion | MultiPoolTablespaceLongRegion"

logger = logging.getLogger(__name__)


# Priority ordering for eviction. Higher number = retained longer.
# `ephemeral` (10) sits between `none` (0, never store) and `short`
# (50). The ordering means a flood of ephemeral writes can't displace
# a short-retention block, but ephemeral entries are always evicted
# before short ones when capacity pressure forces a choice.
_PRIORITY = {
    RETENTION_NONE: 0,
    RETENTION_EPHEMERAL: 10,
    RETENTION_SHORT: 50,
    RETENTION_LONG: 100,
}


def retention_priority(retention: str) -> int:
    """Numeric priority for a retention level. Unknown levels raise."""
    if retention not in _PRIORITY:
        raise ValueError(f"unknown retention: {retention!r}")
    return _PRIORITY[retention]


@dataclass
class Entry:
    """One stored block. Mutable so we can refresh `last_access` cheaply.

    Storage:
    - `value`: the bytes themselves when the store is in inline mode
      (no shared arena wired). Stays as the canonical storage path
      for backward compatibility — existing tests + ops tooling
      access `entry.value` directly.
    - `slot_id`: when the store has a SharedArena wired, the bytes
      live in the arena and `slot_id` is the index. `value` becomes
      a sentinel `b""` and callers must go through `HostStore.resolve_value`
      to read the actual bytes (which materialize a fresh `bytes`
      copy via the arena's seqlock — for non-shared-arena clients
      receiving inline-bytes responses).
    - `size`: the logical byte size of the stored blob. Mirrors
      `len(value)` in inline mode; in arena-backed mode it's the
      authoritative size since `value` is empty.
    """

    key: bytes
    value: bytes
    retention: str
    model: str = ""
    compat_key: str = ""
    metadata: dict = field(default_factory=dict)
    last_access: float = field(default_factory=time.monotonic)
    # Absolute monotonic deadline (`time.monotonic()` units). `None`
    # means "no TTL." If set and `time.monotonic() >= expires_at`,
    # the entry is treated as missing by `get`/`exists` and removed
    # opportunistically (lazy expiration — no sweeper thread). The
    # SET path computes this from `ttl_seconds` so the caller can
    # think in seconds without worrying about clock semantics.
    expires_at: float | None = None
    # Shared-arena slot index. -1 means "inline storage in `value`".
    # >=0 means "bytes live in the arena at this slot index; `value`
    # is empty bytes." Set by HostStore's `set` path when the store
    # was constructed with a `shared_arena`.
    slot_id: int = -1
    # Cached size when slot_id >= 0 (since `value` is empty in that
    # mode). Set to `len(value)` in inline mode for symmetry; either
    # way, `size_bytes` always reports the right thing.
    _size_cache: int = 0

    @property
    def size_bytes(self) -> int:
        if self.slot_id >= 0:
            return self._size_cache
        return len(self.value)

    @property
    def priority(self) -> int:
        return _PRIORITY[self.retention]

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.expires_at


@dataclass
class StoreStats:
    """Counters reported via STATS wire message and Prometheus."""

    entries: int = 0
    host_bytes: int = 0
    gets_total: int = 0
    sets_total: int = 0
    hits_total: int = 0
    misses_total: int = 0
    evictions_total: int = 0
    set_rejections_total: int = 0


class HostStore:
    """Thread-safe in-memory KV store with priority-aware LRU.

    The daemon wraps a single `HostStore` instance per node. All async
    handlers serialize on this store's internal lock — the operations
    are short (hash lookup, bytes assignment) so contention is minimal
    even at high QPS. If profile says otherwise we can shard later by
    `(model, compat_key)`.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        spillover: SpilloverRegion | None = None,
        long_region: LongStorageRegion | TablespaceLongRegion | None = None,
        shared_arena: SharedArena | None = None,
    ) -> None:
        """`max_bytes` is the host RAM budget. SSD regions are optional —
        when None, the store behaves like Phase 3.0 (RAM-only). When
        wired, the store coordinates a 3-tier flow:

            SET retention=long  → host RAM + long region (write_through)
            SET retention=short → host RAM only (lazy)
            RAM eviction of short → write to spillover
            RAM eviction of long  → drop (already on long SSD)
            GET → host RAM → long region → spillover region → miss

        `shared_arena`: optional `SharedArena` (memfd-backed, see
        `infera.kvd.shared_arena`). When provided, the store's host
        RAM tier is backed by the arena instead of an in-Python bytes
        dict. Entries carry a `slot_id` instead of `value` bytes;
        vLLM/SGLang workers that opted into the shared-arena handshake
        receive `(slot_offset, length, version)` on the wire and read
        bytes directly from their own mmap. Inline-bytes clients still
        work — the server materializes bytes via `arena.get_slice`
        on demand."""
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._entries: dict[tuple[str, str, bytes], Entry] = {}
        # Reverse index from raw key-bytes → set of composites that share
        # those bytes (different `(model, compat_key)` namespaces can
        # legitimately map to the same key bytes). Maintained in lockstep
        # with `_entries` mutations under `_lock`. Used by the post-put
        # arena-eviction drain so we don't have to O(N)-scan `_entries`
        # on every set that triggers an arena LRU eviction.
        self._key_index: dict[bytes, set[tuple[str, str, bytes]]] = {}
        self._lock = threading.Lock()
        self._shared_arena = shared_arena
        # Separate lock guarding the long-region write path. Concurrent
        # SETs to the SAME key would otherwise mutate RAM in one order
        # but write SSD in another (PR #9 review fix P0-2). Coarse —
        # all long-region writes serialize on this. For multi-key
        # parallelism we could shard per-composite-hash, but profiling
        # would have to justify the complexity.
        self._long_write_lock = threading.Lock()
        self._stats = StoreStats()
        self._spillover = spillover
        self._long_region = long_region

        # If the long region recovered entries from manifest on startup,
        # adopt their keys into our "known on disk" index so GET can find
        # them. Bytes stay on disk until first GET demands a promotion.
        # (Skipped if long_region is None.)

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def shared_arena(self) -> SharedArena | None:
        """Read-only access to the wired arena, or None when the
        store is inline-bytes only. Server uses this to construct
        the per-connection shared-arena dispatch path."""
        return self._shared_arena

    def resolve_value(self, entry: Entry) -> bytes:
        """Return the entry's bytes, regardless of storage mode.

        Inline mode: returns `entry.value` directly (zero overhead).
        Arena mode: reads via `arena.get_slice(entry.slot_id)` and
        returns the materialized bytes. The server uses this on the
        inline-bytes response path so non-shared-arena clients still
        get their value field populated correctly.

        Returns `b""` if the entry's slot has been evicted from the
        arena (race condition that should not normally happen since
        HostStore controls both lifetimes). Treat as miss in callers.
        """
        if entry.slot_id < 0:
            return entry.value
        if self._shared_arena is None:
            # Shouldn't happen — slot_id was set so an arena was
            # wired at the time. Defensive return.
            return b""
        mv = self._shared_arena.get_slice(entry.slot_id)
        if mv is None:
            return b""
        return bytes(mv)

    @property
    def stats(self) -> StoreStats:
        with self._lock:
            # `host_bytes` reflects the bytes of DATA stored — sum of
            # blob sizes regardless of where they live (inline value
            # field or arena slot). We sum `_size_cache` for arena
            # entries (= original blob length) so the number reflects
            # user-meaningful payload bytes, not the allocator's
            # slot_size padding. `_used_bytes` only tracks inline
            # (post-2026-05-26 fix avoiding double-counting against
            # max_bytes).
            arena_bytes = 0
            if self._shared_arena is not None:
                for e in self._entries.values():
                    if e.slot_id >= 0:
                        arena_bytes += e._size_cache
            return StoreStats(
                entries=len(self._entries),
                host_bytes=self._used_bytes + arena_bytes,
                gets_total=self._stats.gets_total,
                sets_total=self._stats.sets_total,
                hits_total=self._stats.hits_total,
                misses_total=self._stats.misses_total,
                evictions_total=self._stats.evictions_total,
                set_rejections_total=self._stats.set_rejections_total,
            )

    @property
    def spillover_bytes(self) -> int:
        return self._spillover.used_bytes if self._spillover is not None else 0

    @property
    def long_bytes(self) -> int:
        return self._long_region.used_bytes if self._long_region is not None else 0

    # ------------------------------------------------------------------
    # Public API for the prefetch worker
    # ------------------------------------------------------------------
    # The kvd prefetch worker used to reach into `_lock`, `_entries`,
    # `_long_region`, `_spillover` directly. That broke encapsulation
    # AND held the store lock across the long-region's `get_bytes()`
    # disk I/O — every disk read blocked every other get/set on the
    # daemon. These two APIs replace those private-attr reads:
    # `peek_in_ram` is a lock-bounded membership check, and
    # `warm_from_ssd` runs the disk-I/O step OUTSIDE the lock and
    # only re-acquires to insert.

    def peek_in_ram(self, model: str, compat_key: str, key: bytes) -> bool:
        """Return True if the composite `(model, compat_key, key)` is
        currently in host RAM (`_entries`). Does NOT touch LRU
        recency and does NOT consult SSD tiers — it's the membership
        check the prefetch worker needs to skip the no-op case where
        the router over-emits hints for a block we already have.

        The check is performed under the store lock so the read is
        consistent, but the lock is released before this method
        returns.
        """
        composite = (model, compat_key, key)
        with self._lock:
            return composite in self._entries

    def warm_from_ssd(
        self,
        model: str,
        compat_key: str,
        key: bytes,
        ttl_seconds: float,
    ) -> bool:
        """Atomically: if the composite isn't already in RAM, pull
        its bytes from the long region (and then the spillover region
        on miss), then `set()` it back into RAM with the given TTL.
        Returns True iff the warm actually happened.

        Disk I/O (the `get_bytes` calls on the SSD tiers) runs
        OUTSIDE the store lock — releasing the lock during the disk
        read was the second motivation for hoisting this method up
        (the prior implementation held `_lock` across `get_bytes()`,
        so every prefetch warm blocked every other get/set on the
        daemon for the duration of the SSD seek).

        The final RAM insert goes via the public `set()`, so the
        priority/eviction logic and spillover/long-region
        coordination matches normal SETs exactly.
        """
        # Step 1 — membership check under the lock. If the block is
        # already in RAM, the prefetch is a no-op.
        if self.peek_in_ram(model, compat_key, key):
            return False

        # Step 2 — disk fetch with NO lock held. Try long region
        # first (write-through tier — likely contains
        # long-retention blocks), fall back to spillover.
        value: bytes | None = None
        retention_from_l3: str = RETENTION_SHORT
        if self._long_region is not None:
            ssd_entry = self._long_region.get_entry(key, model=model, compat_key=compat_key)
            if ssd_entry is not None:
                value = self._long_region.get_bytes(key, model=model, compat_key=compat_key)
                retention_from_l3 = ssd_entry.retention
        if value is None and self._spillover is not None:
            ssd_entry = self._spillover.get_entry(key, model=model, compat_key=compat_key)
            if ssd_entry is not None:
                value = self._spillover.get_bytes(key, model=model, compat_key=compat_key)
                retention_from_l3 = ssd_entry.retention

        if value is None:
            return False

        # Step 3 — re-acquire the lock implicitly via set(). The
        # composite may have been inserted by another thread between
        # peek_in_ram and now; `set()` will just refresh it. Either
        # way, the block lands in RAM. We sanitize 'none' retention
        # so the RAM tier doesn't reject the warm (a 'none' SSD
        # retention level shouldn't translate to a 'never store in
        # RAM' decision at warm time).
        accepted, _reason = self.set(
            key,
            value,
            retention=retention_from_l3 if retention_from_l3 != RETENTION_NONE else RETENTION_SHORT,
            model=model,
            compat_key=compat_key,
            ttl_seconds=max(ttl_seconds, 0.001),
        )
        return accepted

    def get(self, key: bytes, *, model: str = "", compat_key: str = "") -> Entry | None:
        """Look up a block. Refresh `last_access` on hit (this is what
        makes the LRU work). If the block isn't in RAM, fall through to
        the long region and then the spillover region (when wired). On
        SSD hit, promote the bytes back into RAM so the next access is
        fast.
        """
        composite = (model, compat_key, key)
        with self._lock:
            self._stats.gets_total += 1
            entry = self._entries.get(composite)
            if entry is not None:
                # Lazy TTL expiration. If the entry's deadline has
                # passed, drop it as if it were never there. This
                # avoids the cost of a sweeper thread — TTLs are
                # only enforced when someone actually asks for the
                # block, which matches the cache's access pattern.
                if entry.is_expired():
                    self._remove_entry_locked(composite, entry)
                    self._stats.misses_total += 1
                    return None
                entry.last_access = time.monotonic()
                self._stats.hits_total += 1
                return entry

        # RAM miss — try SSD tiers.
        # Long region first (we check it before spillover because long
        # blocks are higher value and likely retained longer; ordering
        # is functional only — both regions are disjoint so a key can
        # exist in at most one).
        if self._long_region is not None:
            ssd_entry = self._long_region.get_entry(key, model=model, compat_key=compat_key)
            if ssd_entry is not None:
                value = self._long_region.get_bytes(key, model=model, compat_key=compat_key)
                if value is not None:
                    promoted = self._promote_from_ssd(
                        key,
                        value,
                        retention=ssd_entry.retention,
                        model=model,
                        compat_key=compat_key,
                        metadata=ssd_entry.metadata,
                    )
                    if promoted is not None:
                        with self._lock:
                            self._stats.hits_total += 1
                        return promoted

        if self._spillover is not None:
            ssd_entry = self._spillover.get_entry(key, model=model, compat_key=compat_key)
            if ssd_entry is not None:
                value = self._spillover.get_bytes(key, model=model, compat_key=compat_key)
                if value is not None:
                    promoted = self._promote_from_ssd(
                        key,
                        value,
                        retention=ssd_entry.retention,
                        model=model,
                        compat_key=compat_key,
                        metadata=ssd_entry.metadata,
                    )
                    if promoted is not None:
                        with self._lock:
                            self._stats.hits_total += 1
                        return promoted

        with self._lock:
            self._stats.misses_total += 1
        return None

    def exists(self, keys: list[bytes], *, model: str = "", compat_key: str = "") -> list[bool]:
        """Bulk membership test. Doesn't refresh last_access — caller
        is probing, not using the block. Considers SSD tiers when
        wired so the router's cache-locality signal sees them too.

        TTL-expired entries report as missing (and are evicted
        opportunistically) so the router doesn't waste a load on a
        block that will get dropped on the next `get`."""
        with self._lock:
            now = time.monotonic()
            ram_present = []
            for k in keys:
                composite = (model, compat_key, k)
                entry = self._entries.get(composite)
                if entry is None:
                    ram_present.append(False)
                elif entry.is_expired(now):
                    self._remove_entry_locked(composite, entry)
                    ram_present.append(False)
                else:
                    ram_present.append(True)

        # Short-circuit: if everything is in RAM, no need to ask SSD.
        if all(ram_present):
            return ram_present

        # Patch in SSD-resident keys.
        if self._long_region is not None:
            long_present = self._long_region.exists(keys, model=model, compat_key=compat_key)
            ram_present = [a or b for a, b in zip(ram_present, long_present, strict=True)]
        if self._spillover is not None and not all(ram_present):
            spill_present = self._spillover.exists(keys, model=model, compat_key=compat_key)
            ram_present = [a or b for a, b in zip(ram_present, spill_present, strict=True)]
        return ram_present

    def get_many(
        self, keys: list[bytes], *, model: str = "", compat_key: str = ""
    ) -> list[Entry | None]:
        """Batch GET: like `get` for many keys, but reads the long region in
        ONE `get_bytes_batch` call instead of N single `get_bytes`. The win
        is on a distributed L4 backend, where each single get is a full
        network round trip — a multi-block read collapses N RTTs into one.
        Order-preserving; refreshes `last_access` on RAM hits and promotes
        SSD/L4 hits back into RAM, exactly like `get`.

        Promotion note: blocks brought in via the batched long-region read
        carry `retention='long'` (the long region only ever stores
        long-retention blocks) and empty metadata. For the L4 adapters that
        is byte-identical to `get`/`get_entry` (their metadata is always
        {}); for the local L3 tablespace it drops the stored metadata on the
        promoted RAM copy — server-internal only (never returned to clients;
        only matters if that copy is later re-spilled). The spillover tier
        keeps the exact per-key `get_entry` semantics (it's local, so the
        RTT win doesn't apply).
        """
        results: list[Entry | None] = [None] * len(keys)
        miss_idx: list[int] = []
        now = time.monotonic()
        with self._lock:
            for i, k in enumerate(keys):
                self._stats.gets_total += 1
                composite = (model, compat_key, k)
                entry = self._entries.get(composite)
                if entry is not None and not entry.is_expired(now):
                    entry.last_access = now
                    self._stats.hits_total += 1
                    results[i] = entry
                elif entry is not None:
                    self._remove_entry_locked(composite, entry)
                    miss_idx.append(i)
                else:
                    miss_idx.append(i)

        # Long region — ONE batched read for every RAM miss when the region
        # supports it; otherwise the exact per-key get_entry+get_bytes path.
        if miss_idx and self._long_region is not None:
            miss_keys = [keys[i] for i in miss_idx]
            batch_fn = getattr(self._long_region, "get_bytes_batch", None)
            vals: list[bytes | None] | None = None
            if batch_fn is not None:
                try:
                    got = batch_fn(miss_keys, model=model, compat_key=compat_key)
                    if len(got) == len(miss_keys):
                        vals = got
                except Exception:
                    logger.exception("get_many: long-region batch read failed; per-key fallback")
            still_miss: list[int] = []
            if vals is not None:
                # Batched fast path. retention='long' (long region contract);
                # metadata dropped (see docstring) — byte-identical for L4.
                for j, i in enumerate(miss_idx):
                    value = vals[j]
                    if value is None:
                        still_miss.append(i)
                        continue
                    promoted = self._promote_from_ssd(
                        keys[i],
                        value,
                        retention=RETENTION_LONG,
                        model=model,
                        compat_key=compat_key,
                        metadata=None,
                    )
                    if promoted is not None:
                        with self._lock:
                            self._stats.hits_total += 1
                        results[i] = promoted
                    else:
                        still_miss.append(i)
            else:
                # No batch support (legacy region) or batch failed — exact
                # old per-key semantics, preserving retention + metadata.
                for i in miss_idx:
                    k = keys[i]
                    ssd_entry = self._long_region.get_entry(k, model=model, compat_key=compat_key)
                    if ssd_entry is None:
                        still_miss.append(i)
                        continue
                    value = self._long_region.get_bytes(k, model=model, compat_key=compat_key)
                    if value is None:
                        still_miss.append(i)
                        continue
                    promoted = self._promote_from_ssd(
                        k,
                        value,
                        retention=ssd_entry.retention,
                        model=model,
                        compat_key=compat_key,
                        metadata=ssd_entry.metadata,
                    )
                    if promoted is not None:
                        with self._lock:
                            self._stats.hits_total += 1
                        results[i] = promoted
                    else:
                        still_miss.append(i)
            miss_idx = still_miss

        # Spillover — local tier, keep exact per-key semantics.
        if miss_idx and self._spillover is not None:
            resolved: list[int] = []
            for i in miss_idx:
                k = keys[i]
                ssd_entry = self._spillover.get_entry(k, model=model, compat_key=compat_key)
                if ssd_entry is None:
                    continue
                value = self._spillover.get_bytes(k, model=model, compat_key=compat_key)
                if value is None:
                    continue
                promoted = self._promote_from_ssd(
                    k,
                    value,
                    retention=ssd_entry.retention,
                    model=model,
                    compat_key=compat_key,
                    metadata=ssd_entry.metadata,
                )
                if promoted is not None:
                    with self._lock:
                        self._stats.hits_total += 1
                    results[i] = promoted
                    resolved.append(i)
            miss_idx = [i for i in miss_idx if i not in resolved]

        if miss_idx:
            with self._lock:
                self._stats.misses_total += len(miss_idx)
        return results

    def set(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str = RETENTION_SHORT,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[bool, str | None]:
        """Insert or update one block.

        Returns ``(accepted, reason)``. If `accepted=False`, the store
        refused to displace higher-priority blocks. The caller's only
        actionable choice is to downgrade retention or accept the miss.

        SSD coordination:
        - retention=long → after writing RAM, also write_through to the
          long region (durable + persisted across restart).
        - retention=short / none → RAM only at SET time. The spillover
          region picks up the block lazily when it's evicted from RAM.

        ``ttl_seconds``: optional time-to-live in seconds (issue #20
        item 1). When set, the entry is treated as expired by `get` /
        `exists` once ``time.monotonic() - SET_time >= ttl_seconds``,
        regardless of retention class. Useful for prompt-cache style
        clients that know the session bounds. Negative or zero values
        expire immediately (treated as no-op write)."""
        validate_retention(retention)
        expires_at: float | None = None
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                # Caller said "expire immediately" — treat as never-stored.
                # Counts as a SET but no entry is created.
                with self._lock:
                    self._stats.sets_total += 1
                return True, None
            expires_at = time.monotonic() + ttl_seconds
        composite = (model, compat_key, key)
        # PR #9 review fix P0-2 (concurrent-PUT RAM↔SSD race):
        # - `spill_pending` MUST be a local variable. Stashing it on
        #   `self` like the previous code did meant two concurrent
        #   `set()` calls would overwrite each other's pending list,
        #   silently dropping spillover writes.
        # - The long-region write happens under `_long_write_lock`
        #   AFTER re-reading the latest RAM value — so two concurrent
        #   writers to the SAME key serialize on disk and the
        #   *latest* RAM state is what lands on SSD. Without this,
        #   thread A could write its RAM value first then thread B
        #   could overwrite RAM-then-call-put, leaving RAM=B and
        #   SSD=A.
        spill_pending: list[tuple[bytes, bytes, str, str, dict]] = []
        with self._lock:
            self._stats.sets_total += 1

            existing = self._entries.get(composite)
            if existing is not None:
                if _PRIORITY[retention] < existing.priority:
                    self._stats.set_rejections_total += 1
                    return False, "retention_downgrade_not_allowed"
                # `_used_bytes` only tracks INLINE-stored bytes. Arena-
                # backed entries (slot_id >= 0) have their bytes in the
                # arena's own slot-count cap and must NOT be counted
                # here — counting them double-billed RAM accounting
                # against max_bytes vs arena capacity, causing premature
                # eviction storms (see bench notes 2026-05-26).
                old_inline = 0 if existing.slot_id >= 0 else len(existing.value)
                if self._shared_arena is not None:
                    # Push new bytes into the arena. The arena's
                    # `put` returns the slot index (may reuse the
                    # old slot since `put` first frees on overwrite).
                    new_slot = self._shared_arena.put(key, value)
                    if new_slot is None:
                        # Arena rejected (size mismatch). Fall back
                        # to inline storage for this entry — a mixed
                        # store is fine; the GET path branches on
                        # `slot_id >= 0`.
                        existing.value = value
                        existing.slot_id = -1
                        existing._size_cache = 0
                    else:
                        existing.value = b""
                        existing.slot_id = new_slot
                        existing._size_cache = len(value)
                else:
                    existing.value = value
                existing.retention = retention
                existing.last_access = time.monotonic()
                existing.metadata = metadata or {}
                # Refresh TTL on overwrite — caller's intent is "set
                # this block again with these semantics," including a
                # fresh expiration deadline.
                existing.expires_at = expires_at
                # Apply the inline-bytes delta NOW that the entry's
                # post-overwrite state is known. If the entry stayed
                # arena-backed (slot_id >= 0), the new inline-size is
                # 0. If it fell back to inline (arena rejected), new
                # inline-size is len(value).
                new_inline = 0 if existing.slot_id >= 0 else len(existing.value)
                self._used_bytes += new_inline - old_inline
            else:
                incoming_size = len(value)
                # Predict whether the incoming entry will go to arena
                # (arena handles its own capacity) vs land inline
                # (counted against max_bytes). If arena is wired AND
                # the size would fit a slot, this set won't add inline
                # bytes — skip the eviction loop entirely.
                would_use_arena = (
                    self._shared_arena is not None
                    and self._shared_arena.would_accept_size(incoming_size)
                )
                inline_demand = 0 if would_use_arena else incoming_size
                if inline_demand > self._max_bytes:
                    self._stats.set_rejections_total += 1
                    return False, "value_larger_than_store"

                # Make room. Evict in priority order (lowest first), then LRU.
                # On eviction we collect short-retention victims and write them
                # to the spillover region AFTER releasing the lock (disk I/O
                # is slow; we don't want to block the daemon).
                # `inline_demand` is 0 when the entry will go to the arena
                # — the loop short-circuits, and the arena's own slot LRU
                # handles capacity for arena-backed bytes.
                while self._used_bytes + inline_demand > self._max_bytes:
                    victim = self._pick_victim(incoming_retention=retention)
                    if victim is None:
                        self._stats.set_rejections_total += 1
                        return False, "would_displace_higher_priority"
                    spill_payload = self._collect_for_spillover_locked(victim)
                    self._evict_locked(victim)
                    if spill_payload is not None:
                        spill_pending.append(spill_payload)

                # If a shared arena is wired, the bytes live there;
                # the Entry only carries a slot_id (and value=b"")
                # to keep the in-Python overhead constant regardless
                # of blob size.
                if self._shared_arena is not None:
                    slot = self._shared_arena.put(key, value)
                    if slot is None:
                        # Arena rejection (e.g. size mismatch with
                        # already-fixed slot_size). Fall back to inline.
                        entry = Entry(
                            key=key,
                            value=value,
                            retention=retention,
                            model=model,
                            compat_key=compat_key,
                            metadata=metadata or {},
                            expires_at=expires_at,
                        )
                    else:
                        entry = Entry(
                            key=key,
                            value=b"",
                            retention=retention,
                            model=model,
                            compat_key=compat_key,
                            metadata=metadata or {},
                            expires_at=expires_at,
                            slot_id=slot,
                            _size_cache=incoming_size,
                        )
                else:
                    entry = Entry(
                        key=key,
                        value=value,
                        retention=retention,
                        model=model,
                        compat_key=compat_key,
                        metadata=metadata or {},
                        expires_at=expires_at,
                    )
                self._entries[composite] = entry
                self._key_index_add(composite)
                # Only inline-stored bytes count against max_bytes.
                # Arena-backed entries (slot_id >= 0) live in the
                # arena's separate slot grid and must not be billed
                # against the host-RAM cap.
                if entry.slot_id < 0:
                    self._used_bytes += incoming_size

            # Drain any keys the arena's own LRU evicted during the
            # put above. Without this, HostStore's `_entries` would
            # still hold entries pointing at recycled slots, and
            # subsequent `get` would return the wrong key's bytes.
            # `_key_index` gives O(1) key-bytes → composites lookup
            # so the drain is O(evicted) instead of O(entries) (we
            # used to rebuild a full reverse map on every set that
            # triggered an eviction; that scan was the dominant
            # hot-path cost under sustained arena pressure).
            #
            # Two composites can legitimately share the same key
            # bytes (different `(model, compat_key)` namespaces) —
            # the index uses set semantics so we drop every entry
            # pointing at the freed slot, not just one of them.
            if self._shared_arena is not None:
                evicted = self._shared_arena.drain_recent_evictions()
                if evicted:
                    for ek in evicted:
                        comps = self._key_index.get(ek)
                        if not comps:
                            continue
                        # Snapshot the bucket — we mutate
                        # `_key_index` via `_key_index_remove` inside
                        # the loop, which would invalidate iteration.
                        for ec in list(comps):
                            ev_entry = self._entries.pop(ec, None)
                            if ev_entry is None:
                                continue
                            self._key_index_remove(ec)
                            if ev_entry.slot_id < 0:
                                self._used_bytes -= ev_entry.size_bytes
                            # The arena already freed the slot — no need
                            # to call evict_key again.
                            self._stats.evictions_total += 1

        # Now outside the RAM lock — do SSD I/O.
        # 1. Flush any spillover writes from RAM evictions.
        if spill_pending and self._spillover is not None:
            for k, v, model_, compat_key_, meta_ in spill_pending:
                self._spillover.put(
                    k,
                    v,
                    retention=RETENTION_SHORT,
                    model=model_,
                    compat_key=compat_key_,
                    metadata=meta_,
                )

        # 2. Write_through to long region if this SET is long-retention.
        if retention == RETENTION_LONG and self._long_region is not None:
            # Serialize all long writes — concurrent writers to the
            # SAME key would otherwise land on SSD in an order
            # different from how they hit RAM. Inside this lock we
            # re-read the current RAM value (it may have moved
            # forward since our own RAM mutation) so SSD ends up
            # holding whatever RAM has at this moment.
            with self._long_write_lock:
                with self._lock:
                    current = self._entries.get(composite)
                    if current is None:
                        # Evicted between RAM mutation and SSD write —
                        # don't propagate stale bytes to SSD. The RAM
                        # eviction path would have spilled via the
                        # spillover region already.
                        return True, None
                    # Resolve the entry's bytes (materializing from the
                    # arena if arena-backed) under the RAM lock so the
                    # slot can't be evicted under us while we read.
                    current_metadata = dict(current.metadata)
                    current_value = self.resolve_value(current)
                accepted_long: bool
                reason_long: str | None
                accepted_long, reason_long = self._long_region.put(
                    key,
                    current_value,
                    retention=RETENTION_LONG,
                    model=model,
                    compat_key=compat_key,
                    metadata=current_metadata,
                )
                if not accepted_long:
                    logger.warning(
                        "long region rejected long-retention SET key=%s reason=%s — "
                        "block lives only in RAM until next eviction",
                        key.hex()[:16],
                        reason_long,
                    )

        return True, None

    def commit_arena_lease(
        self,
        composite: tuple[str, str, bytes],
        slot_id: int,
        length: int,
        *,
        retention: str = RETENTION_SHORT,
        ttl_seconds: float | None = None,
        overwritten_key: bytes | None = None,
    ) -> None:
        """Insert (or replace) the entry for `composite` after the
        arena has already accepted bytes into `slot_id`.

        Used by the zero-copy save path: the server has called
        `arena.commit_reservation(...)`, the slot is now holding
        valid bytes with a stable seqlock version, and we just need
        to make the HostStore index point at it.

        `overwritten_key` is the key (raw bytes) whose previous arena
        slot the commit freed because the new key collided with it.
        When present, we drop every composite in `_entries` that
        shared those key bytes — the arena has already released
        their slot, so leaving them in `_entries` would point at a
        recycled slot. Mirrors the arena-eviction drain in `set()`.

        Counters:
        - `sets_total` bumps (so observability is symmetric with the
          legacy `set()` path).
        - Inline `_used_bytes` is NOT touched — arena-backed entries
          don't bill against `max_bytes`.
        """
        validate_retention(retention)
        expires_at: float | None = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._stats.sets_total += 1

            # Drop any composite-rows for the freed-by-collision key.
            if overwritten_key is not None:
                bucket = self._key_index.get(overwritten_key)
                if bucket:
                    for ec in list(bucket):
                        ev_entry = self._entries.pop(ec, None)
                        if ev_entry is None:
                            continue
                        self._key_index_remove(ec)
                        if ev_entry.slot_id < 0:
                            self._used_bytes -= ev_entry.size_bytes
                        # Don't call evict_key — the arena already
                        # freed the slot via commit_reservation's
                        # overwrite path.

            existing = self._entries.get(composite)
            if existing is not None:
                # Refresh the existing row to point at the new slot.
                # If the previous entry was inline, release its inline
                # byte accounting.
                if existing.slot_id < 0:
                    self._used_bytes -= existing.size_bytes
                existing.value = b""
                existing.slot_id = slot_id
                existing._size_cache = length
                existing.retention = retention
                existing.last_access = time.monotonic()
                existing.expires_at = expires_at
            else:
                key_bytes = composite[2]
                entry = Entry(
                    key=key_bytes,
                    value=b"",
                    retention=retention,
                    model=composite[0],
                    compat_key=composite[1],
                    metadata={},
                    expires_at=expires_at,
                    slot_id=slot_id,
                    _size_cache=length,
                )
                self._entries[composite] = entry
                self._key_index_add(composite)

            # Drain arena-evicted keys recorded during the underlying
            # reserve. Mirrors the post-put dance in `set()` — without
            # this an LRU victim's row would point at a recycled slot.
            if self._shared_arena is not None:
                evicted = self._shared_arena.drain_recent_evictions()
                if evicted:
                    for ek in evicted:
                        comps = self._key_index.get(ek)
                        if not comps:
                            continue
                        for ec in list(comps):
                            ev_entry = self._entries.pop(ec, None)
                            if ev_entry is None:
                                continue
                            self._key_index_remove(ec)
                            if ev_entry.slot_id < 0:
                                self._used_bytes -= ev_entry.size_bytes
                            self._stats.evictions_total += 1

    def clear(self, *, model: str = "", compat_key: str = "") -> int:
        """Drop all entries matching `(model, compat_key)`. Pass empty
        strings (default) to clear the whole store, including SSD tiers.
        Returns the count evicted (RAM only — SSD evictions don't add
        to the count to keep the number meaningful for ops dashboards)."""
        with self._lock:
            if not model and not compat_key:
                count = len(self._entries)
                # Free arena slots before dropping the entries (so
                # the arena's LRU/free-list stays consistent).
                if self._shared_arena is not None:
                    for e in list(self._entries.values()):
                        if e.slot_id >= 0:
                            self._shared_arena.evict_key(e.key, notify_drain=False)
                self._entries.clear()
                self._key_index.clear()
                self._used_bytes = 0
            else:
                to_drop = [
                    composite
                    for composite in self._entries
                    if composite[0] == model and composite[1] == compat_key
                ]
                for composite in to_drop:
                    entry = self._entries.pop(composite)
                    self._key_index_remove(composite)
                    if entry.slot_id < 0:
                        self._used_bytes -= entry.size_bytes
                    if entry.slot_id >= 0 and self._shared_arena is not None:
                        self._shared_arena.evict_key(entry.key, notify_drain=False)
                count = len(to_drop)

        # SSD tiers cleared outside the lock — file deletion is slow.
        # Clear-by-namespace pushes through to SSD too; clear-all wipes
        # both regions entirely.
        if self._spillover is not None:
            if not model and not compat_key:
                self._spillover.clear()
            else:
                self._clear_ssd_namespace(self._spillover, model, compat_key)
        if self._long_region is not None:
            if not model and not compat_key:
                self._long_region.clear()
            else:
                self._clear_ssd_namespace(self._long_region, model, compat_key)
        return count

    @staticmethod
    def _clear_ssd_namespace(region, model: str, compat_key: str) -> None:
        """Remove all entries from `region` matching (model, compat_key).

        Dispatch order:
          1. Public `clear_namespace(model, compat_key)` if the region
             provides it. Multipool needs this — it has no `_entries`
             attr (PR #9 review fix P1).
          2. Fall back to iterating the private `_entries` attr (the
             original single-pool TablespaceLongRegion / file-per-block
             LongStorageRegion path).
        """
        clear_ns = getattr(region, "clear_namespace", None)
        if callable(clear_ns):
            clear_ns(model, compat_key)
            return
        # Legacy / single-pool fallback — iterate the private index.
        try:
            entries_snapshot = list(region._entries.items())  # access internal index
        except AttributeError:
            logger.warning(
                "clear_namespace on %s: no public method and no _entries "
                "attribute; namespace clear is a no-op",
                type(region).__name__,
            )
            return
        for (m, ck, k), _ in entries_snapshot:
            if m == model and ck == compat_key:
                region.remove(k, model=m, compat_key=ck)

    def iter_entries(self) -> Iterator[Entry]:
        """Snapshot iterator (copies the entry list under lock). Used by
        tests and the future SSD spillover background task."""
        with self._lock:
            return iter(list(self._entries.values()))

    # ------------------------------------------------------------------
    # Internal — must be called with `self._lock` held
    # ------------------------------------------------------------------

    def _pick_victim(self, *, incoming_retention: str) -> tuple[str, str, bytes] | None:
        """Return the composite key of the best eviction candidate, or
        None if no entry is at-or-below the incoming priority.

        "Best" = lowest priority, then least-recently-accessed.
        """
        incoming_priority = _PRIORITY[incoming_retention]
        # Single pass — store sizes are bounded by the eviction loop so
        # this is fine for v1. A more sophisticated implementation would
        # maintain per-priority LRU heaps.
        best: tuple[str, str, bytes] | None = None
        best_priority = float("inf")
        best_last_access = float("inf")
        for composite, entry in self._entries.items():
            if entry.priority > incoming_priority:
                continue  # higher than us — protected
            if entry.priority < best_priority or (
                entry.priority == best_priority and entry.last_access < best_last_access
            ):
                best = composite
                best_priority = entry.priority
                best_last_access = entry.last_access
        return best

    def _collect_for_spillover_locked(
        self, composite: tuple[str, str, bytes]
    ) -> tuple[bytes, bytes, str, str, dict] | None:
        """Pre-eviction hook: if the victim is a SHORT-retention block
        and a spillover region is wired, snapshot its bytes so we can
        write to disk after releasing the lock. Returns the tuple to
        spill, or None if no spillover (the block just drops on
        eviction).

        Long-retention blocks aren't spilled here because they're
        already on the long region (write_through at SET time).
        """
        if self._spillover is None:
            return None
        entry = self._entries.get(composite)
        if entry is None or entry.retention != RETENTION_SHORT:
            return None
        # Materialize bytes from the arena if arena-backed. We snap
        # before the eviction releases the slot, so the read is
        # against the still-valid slot. Once the snap is in hand,
        # the arena can free the slot — we hold our own bytes copy.
        value_bytes = self.resolve_value(entry)
        return (entry.key, value_bytes, entry.model, entry.compat_key, dict(entry.metadata))

    def _promote_from_ssd(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
    ) -> Entry | None:
        """Bring a block from an SSD tier back into host RAM. Called
        on GET miss when the SSD tiers had the block. Returns the
        newly-cached `Entry`, or None if the RAM SET was rejected
        (in which case the caller should just return the bytes
        synthesized into a one-off Entry without inserting).
        """
        # Insert into RAM. Reuse the regular set() path so we honor
        # eviction + cascade-to-spillover semantics. Returns
        # (accepted, reason); if rejected we don't fail the GET —
        # the SSD bytes are still valid, so synthesize a transient
        # Entry to return to the caller.
        accepted, _reason = self.set(
            key, value, retention=retention, model=model, compat_key=compat_key, metadata=metadata
        )
        if accepted:
            with self._lock:
                return self._entries.get((model, compat_key, key))
        # Rejected (e.g., short retention can't displace higher) —
        # return a one-off Entry for the caller, but DON'T cache.
        return Entry(
            key=key,
            value=value,
            retention=retention,
            model=model,
            compat_key=compat_key,
            metadata=metadata or {},
        )

    def _evict_locked(self, composite: tuple[str, str, bytes]) -> None:
        entry = self._entries.pop(composite)
        self._key_index_remove(composite)
        # `_used_bytes` only tracks inline bytes (arena-backed entries
        # don't count — they live in the arena's separate slot grid).
        # Decrement matches the increment policy in `set()`.
        if entry.slot_id < 0:
            self._used_bytes -= entry.size_bytes
        # If arena-backed, release the slot back to the arena's free
        # list so the next put can reuse it.
        if entry.slot_id >= 0 and self._shared_arena is not None:
            self._shared_arena.evict_key(entry.key, notify_drain=False)
        self._stats.evictions_total += 1
        logger.debug(
            "evicted key=%s retention=%s size=%d",
            entry.key.hex()[:16],
            entry.retention,
            entry.size_bytes,
        )

    def _remove_entry_locked(self, composite: tuple[str, str, bytes], entry: Entry) -> None:
        """Remove an entry without bumping the evictions counter — used
        by lazy TTL expiration where the removal isn't a
        capacity-pressure eviction. Caller is responsible for any
        miss/expired stats accounting."""
        if self._entries.pop(composite, None) is not None:
            self._key_index_remove(composite)
        if entry.slot_id < 0:
            self._used_bytes -= entry.size_bytes
        # Release the arena slot if backed.
        if entry.slot_id >= 0 and self._shared_arena is not None:
            self._shared_arena.evict_key(entry.key, notify_drain=False)

    # ------------------------------------------------------------------
    # Internal — key-bytes → composite reverse index helpers (must be
    # called with `_lock` held; they update `_key_index` in lockstep with
    # `_entries` so the arena-eviction drain is O(1) per evicted key).
    # ------------------------------------------------------------------

    def _key_index_add(self, composite: tuple[str, str, bytes]) -> None:
        key_bytes = composite[2]
        bucket = self._key_index.get(key_bytes)
        if bucket is None:
            self._key_index[key_bytes] = {composite}
        else:
            bucket.add(composite)

    def _key_index_remove(self, composite: tuple[str, str, bytes]) -> None:
        key_bytes = composite[2]
        bucket = self._key_index.get(key_bytes)
        if bucket is None:
            return
        bucket.discard(composite)
        if not bucket:
            del self._key_index[key_bytes]
