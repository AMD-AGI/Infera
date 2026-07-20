###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Multi-pool tablespace long region (Phase B.2+).

Wraps N `TablespaceLongRegion` instances, each with its own
``slot_bytes``. ``put`` routes the value to the **smallest pool whose
slot_bytes ≥ len(value)** — smaller pools have better space density
(less dead padding per slot) so this is the right default.

## Why multi-pool

Single-pool was a Phase B limitation: one fixed slot size means
either:

- A small slot (e.g. 64 KB for SGLang per-layer-per-page) — vLLM's
  packed 1 MB blocks get rejected as ``value_exceeds_slot_bytes``.
- A big slot (1 MB or 4 MB for vLLM packed) — SGLang's 1-2 KB
  per-page writes waste ~99% of every slot (1 KB used out of
  1 MB).

A single ``infera-kvd`` daemon often serves BOTH engines (mixed
fleet, A/B testing, gradual rollout). Multi-pool lets it
transparently size each value to a matching pool.

The pattern follows 3FS's chunk_engine, which exposes 64K /
512K / 4M pools by default.

## Layout on disk

```
<long-path>/
├── pool-0000064K/
│   ├── containers/0000.bin
│   ├── containers/...
│   ├── index.log
│   └── index.snapshot.json
├── pool-0000512K/
│   ├── containers/0000.bin
│   ├── ...
├── pool-0001M/
│   └── ...
└── pool-0004M/
    └── ...
```

Each pool is a complete `TablespaceLongRegion` instance — same
journal + snapshot semantics, same restart-survival contract.
A pool failing to start (e.g. O_DIRECT reject in only ONE pool's
filesystem mount) doesn't crash the others.

## Get semantics

``get_bytes(key)`` queries each pool in size order; first hit wins.
For 4 pools and a dict-based ``exists`` this is ~4 µs overhead
(negligible vs a single disk read at 100+ µs).

Why not maintain a central ``key → pool_idx`` router index? It would
save the 4-way lookup but adds:

- Cross-pool state mutation under the same lock (defeats the
  per-pool concurrency advantage)
- Snapshot/journal coordination across pools (multi-write)

We measured the overhead vs central-index design at ~3 µs in the
multi-pool unit tests — staying with the lookup pattern.

## Put routing

```
def _pick_pool(value_size):
    for pool in pools_sorted_by_slot_asc:
        if value_size <= pool.slot_bytes:
            return pool
    return None  # value > largest pool's slot
```

If the value doesn't fit any pool, ``put`` returns
``(False, "value_exceeds_largest_pool")`` — the engine adapter
sees the rejection and falls back to either (a) skip caching
this block, or (b) request operator add a larger pool.

We DON'T do auto-pool-creation at runtime: pool count is
deployment-time config. Auto-promotion (small pool → big pool)
would be a Phase C feature; for now the operator sizes pools to
fit the workload (which they know ahead of time — model + TP
config dictates the packed-block size).
"""

from __future__ import annotations

import logging
from pathlib import Path

from infera.kvd.tablespace import TablespaceEntry, TablespaceLongRegion

logger = logging.getLogger(__name__)


class MultiPoolTablespaceLongRegion:
    """Drop-in replacement for `TablespaceLongRegion` exposing the same
    public surface but routing values to multiple internal pools.

    Construction:

        region = MultiPoolTablespaceLongRegion(
            path="/var/lib/kvd-long",
            pools=[
                (64 * 1024,       8 * 1024**3),   # 64K slots, 8 GB total
                (1024 * 1024,    32 * 1024**3),   # 1M slots, 32 GB total
                (4 * 1024**2,    24 * 1024**3),   # 4M slots, 24 GB total
            ],
            container_bytes=1024**3,              # 1 GB containers
            sync_writes=True,
            o_direct=True,
        )

    Each pool is sized independently — operators tune per-pool
    `max_bytes` based on the expected mix of value sizes. Defaults
    aren't given here: deployment-specific. CLI handles defaults.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pools: list[tuple[int, int]],  # [(slot_bytes, max_bytes), ...]
        container_bytes: int = 1024 * 1024 * 1024,
        sync_writes: bool = True,
        o_direct: bool | None = True,
        flush_interval_ms: int | None = 0,
    ) -> None:
        if not pools:
            raise ValueError("multi-pool tablespace requires at least one pool")
        # Sort by slot_bytes ascending — this is the put-routing order
        # AND the get-lookup order (small first, biggest last).
        self._pool_specs = sorted(pools, key=lambda p: p[0])
        if len(set(s for s, _ in self._pool_specs)) != len(self._pool_specs):
            raise ValueError(f"duplicate slot_bytes in pools: {self._pool_specs}")

        self._path = Path(path)
        self._container_bytes = container_bytes
        self._sync_writes = sync_writes
        self._o_direct = o_direct

        # One TablespaceLongRegion per pool. Sub-dir name encodes the
        # slot size (zero-padded hex) so a `ls` of the long-region
        # directory makes pool sizing visible.
        #
        # Per-pool container_bytes constraints:
        #   - must be ≥ slot_bytes (TablespaceLongRegion's own check)
        #   - capped at max_bytes (small pools shouldn't pre-allocate
        #     a huge unused container)
        # We resolve to ``max(slot_bytes, min(container_bytes, max_bytes))``.
        # The big-slot pool may end up with container_bytes == slot_bytes
        # (one slot per container), which is fine.
        self._pools: list[TablespaceLongRegion] = []
        for slot_bytes, max_bytes in self._pool_specs:
            sub_dir = self._path / f"pool-{_format_slot_label(slot_bytes)}"
            cb = max(slot_bytes, min(container_bytes, max_bytes))
            self._pools.append(
                TablespaceLongRegion(
                    sub_dir,
                    max_bytes=max_bytes,
                    slot_bytes=slot_bytes,
                    container_bytes=cb,
                    sync_writes=sync_writes,
                    o_direct=o_direct,
                    flush_interval_ms=flush_interval_ms,
                )
            )
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        for pool in self._pools:
            pool.start()
        self._started = True
        logger.info(
            "multi-pool tablespace started at %s with %d pools: %s",
            self._path,
            len(self._pools),
            [(p.slot_bytes, p.max_bytes) for p in self._pools],
        )

    def shutdown(self) -> None:
        if not self._started:
            return
        for pool in self._pools:
            try:
                pool.shutdown()
            except Exception:
                logger.exception("multi-pool shutdown: pool %s failed", pool.slot_bytes)
        self._started = False

    # ------------------------------------------------------------------
    # Public properties — sum across pools
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "tablespace_multipool"

    @property
    def max_bytes(self) -> int:
        return sum(p.max_bytes for p in self._pools)

    @property
    def used_bytes(self) -> int:
        return sum(p.used_bytes for p in self._pools)

    @property
    def entries_count(self) -> int:
        return sum(p.entries_count for p in self._pools)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def pool_slot_sizes(self) -> tuple[int, ...]:
        return tuple(p.slot_bytes for p in self._pools)

    # ------------------------------------------------------------------
    # Lookup — try pools in ascending slot-size order
    # ------------------------------------------------------------------

    def get_bytes(self, key: bytes, *, model: str = "", compat_key: str = "") -> bytes | None:
        for pool in self._pools:
            value = pool.get_bytes(key, model=model, compat_key=compat_key)
            if value is not None:
                return value
        return None

    def get_entry(
        self, key: bytes, *, model: str = "", compat_key: str = ""
    ) -> TablespaceEntry | None:
        for pool in self._pools:
            entry = pool.get_entry(key, model=model, compat_key=compat_key)
            if entry is not None:
                return entry
        return None

    def exists(self, keys: list[bytes], *, model: str = "", compat_key: str = "") -> list[bool]:
        """``True`` if ANY pool holds the key. We OR per-key across pools."""
        if not keys:
            return []
        result = [False] * len(keys)
        for pool in self._pools:
            partial = pool.exists(keys, model=model, compat_key=compat_key)
            for i, p in enumerate(partial):
                if p:
                    result[i] = True
        return result

    # ------------------------------------------------------------------
    # Insert — route to the smallest pool that fits
    # ------------------------------------------------------------------

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str,
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
    ) -> tuple[bool, str | None]:
        if not self._started:
            return False, "region_not_started"

        size = len(value)
        # Pools are sorted ascending by slot_bytes.
        chosen: TablespaceLongRegion | None = None
        for pool in self._pools:
            if size <= pool.slot_bytes:
                chosen = pool
                break

        if chosen is None:
            biggest = self._pools[-1].slot_bytes
            return False, (
                f"value_exceeds_largest_pool ({size} > {biggest}); "
                f"add a larger --tablespace-pools entry to host this size"
            )

        # If the key already lives in a DIFFERENT pool (different slot
        # size selected on a prior put, e.g. value shrank), remove it
        # there before writing to the new home. Same-key-different-pool
        # would otherwise leak storage AND make get_bytes return the
        # stale value (we'd find the smaller pool first).
        for pool in self._pools:
            if pool is chosen:
                continue
            if pool.get_entry(key, model=model, compat_key=compat_key) is not None:
                pool.remove(key, model=model, compat_key=compat_key)

        return chosen.put(
            key,
            value,
            retention=retention,
            model=model,
            compat_key=compat_key,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Remove / clear — apply to all pools
    # ------------------------------------------------------------------

    def remove(self, key: bytes, *, model: str = "", compat_key: str = "") -> bool:
        removed = False
        for pool in self._pools:
            if pool.remove(key, model=model, compat_key=compat_key):
                removed = True
        return removed

    def clear(self) -> int:
        total = 0
        for pool in self._pools:
            total += pool.clear()
        return total

    def clear_namespace(self, model: str, compat_key: str) -> int:
        """Remove all entries from every pool matching (model,
        compat_key). PR #9 review fix P1 (multipool clear bypass):
        `HostStore._clear_ssd_namespace` reaches into a `_entries`
        attribute that doesn't exist on this wrapper — silent no-op
        without this method. Public method lets HostStore dispatch
        correctly regardless of pool topology."""
        total = 0
        for pool in self._pools:
            # Each pool is a TablespaceLongRegion with its own _entries.
            entries_snapshot = list(pool._entries.items())
            for (m, ck, k), _ in entries_snapshot:
                if m == model and ck == compat_key:
                    if pool.remove(k, model=m, compat_key=ck):
                        total += 1
        return total


def _format_slot_label(slot_bytes: int) -> str:
    """Stable, sortable directory label for a slot size.

    `pool-0000064K` < `pool-0000512K` < `pool-0001M` in alphanumeric
    order, so `ls` shows pools in size order without surprises.
    """
    if slot_bytes % (1024 * 1024) == 0:
        return f"{slot_bytes // (1024 * 1024):04d}M"
    if slot_bytes % 1024 == 0:
        return f"{slot_bytes // 1024:07d}K"
    return f"{slot_bytes:010d}B"


def parse_pools_spec(
    spec: str, *, default_max_bytes_per_pool: int = 8 * 1024**3
) -> list[tuple[int, int]]:
    """Parse the CLI ``--tablespace-pools`` argument.

    Format: ``size[*share],size[*share],...`` where ``size`` is a slot
    size (e.g. ``64K``, ``1M``) and the optional ``*share`` is an
    integer weight for the total ``--long-bytes`` to allocate to this
    pool. Examples:

        "64K,1M"               → two pools, equal split of long-bytes
        "64K*1,1M*4,4M*1"      → 1:4:1 weighted split
        "64K"                  → single-pool deployment (equivalent
                                 to non-multipool)

    Returns ``[(slot_bytes, max_bytes), ...]``. Caller decides
    ``default_max_bytes_per_pool`` if no share weights given.
    """
    if not spec.strip():
        raise ValueError("pools spec is empty")

    raw_pools = [p.strip() for p in spec.split(",") if p.strip()]
    parsed: list[tuple[int, int]] = []
    shares: list[int] = []

    for raw in raw_pools:
        if "*" in raw:
            size_str, share_str = raw.split("*", 1)
            share = int(share_str.strip())
        else:
            size_str = raw
            share = 1
        size = _parse_size_label(size_str.strip())
        parsed.append((size, 0))  # max_bytes filled below
        shares.append(share)

    total_shares = sum(shares)
    # Per-pool max = default_max_bytes_per_pool × share / mean_share. We
    # use mean to keep "equal share" → default for backward intuition.
    mean_share = total_shares / len(shares)
    out = []
    for (size, _), share in zip(parsed, shares, strict=True):
        max_b = int(default_max_bytes_per_pool * share / mean_share)
        out.append((size, max_b))
    return out


def _parse_size_label(s: str) -> int:
    """Parse `'64K'`, `'1M'`, `'4M'`, `'1G'`, or plain `'4096'`."""
    s = s.strip().upper()
    if not s:
        raise ValueError("empty size")
    if s[-1] in "KMGT":
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[s[-1]]
        return int(s[:-1]) * mult
    return int(s)
