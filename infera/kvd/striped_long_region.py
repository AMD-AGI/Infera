###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Multi-device striped long region.

Wraps N independent `TablespaceLongRegion` instances, each sitting on its
own NVMe mount point. Keys are hash-routed to a shard via
``hashlib.blake2b`` (NOT Python's salted ``hash()``, which would shuffle
the mapping across kvd restarts and lose every cached block). All writes
for a key go to one shard; reads for the same key go to the same shard.

The win is in ``get_bytes_batch``: keys are grouped by shard, sub-batches
are dispatched to ``workers_per_shard`` independent threads per shard,
and the per-shard ``pread`` storms run in parallel against N kernel
inodes on N different NVMe block devices. Phase 2 (io_uring) only got
0.94× on real NVMe because every IO serialized on the single inode's
lock — N inodes on N devices gives real fanout.

## Why a per-shard sub-pool (workers_per_shard > 1)

With one worker per shard, each shard's batch executes its pread calls
serially inside a single thread. On an 8-NVMe MI355X node with batch=64
(8 keys/shard, ~0.64 ms per 4 MB read), that's 5.12 ms wall per shard —
even with 8 shards running in parallel the aggregate is capped at
~5-6 GB/s, well under the ~50 GB/s NVMe ceiling.

Splitting each shard's batch across W sub-workers turns each shard into
a W-way parallel pread storm. Since TablespaceLongRegion.get_bytes
releases its dict lock before pread (see test_get_bytes_parallelizes_*),
sub-workers actually run concurrently in the kernel.

Measured on an 8-NVMe MI355X node (batch=32, COLD, O_DIRECT, 4 MB blocks):
  wps=1: 4.63 GB/s   (one worker per shard, PR #27 baseline)
  wps=2: 5.65 GB/s
  wps=4: 6.29 GB/s
  wps=8: 9.56 GB/s   <-- knee of curve, default
  wps=16: 10.07 GB/s (marginal improvement)
  wps=32: 10.12 GB/s (saturated, CPU-bound)

The ~10 GB/s ceiling above wps=8 is the Python dispatch overhead —
per-block latency at wps=32 is 0.33 ms, where the actual NVMe pread is
~0.6 ms, meaning each future spends most of its life in Python dispatch
not in the kernel. Lowering that ceiling would need vectored IO or
io_uring; outside the scope of this sub-pool work.

Re-measured on the same node after the per-container hash allocator landed
(``TablespaceLongRegion`` now distributes new slots across container
inodes by ``blake2b(composite) % num_containers``):

  wps=4  bs=32:  5.87 GB/s   (cold, O_DIRECT, 1 GB containers)
  wps=8  bs=32:  9.76 GB/s
  wps=16 bs=32:  10.04 GB/s

Same shape — the curve flattens around 10 GB/s. Hash distribution is
structurally correct (iostat-x confirms all 8 NVMe devices see
non-zero r/s and %util during cold reads, no single inode hot-spots),
but the dispatch ceiling identified above is still the dominant
constraint at this geometry. The win surfaces in cases the bench
doesn't exercise: production deployments where a single shard holds
many more keys than slots-per-container × 8 (= 2K at slot=4M /
container=1G), where the legacy sequential allocator would have all
new PUTs queued at one inode and reads would serialize on the inode
lock. Real production lookup mixes also fan across containers (not
just sequential bench keys), which the new allocator handles
naturally.

## Why a stable hash

The default ``hash(bytes)`` in CPython is salted per-process via
``PYTHONHASHSEED``, so a key that wrote to shard 3 yesterday might map
to shard 5 today — looking it up after a kvd restart would miss
everywhere. Use ``hashlib.blake2b(key, digest_size=8)`` instead: ~250
ns/key on Zen 4, faster than sha256, deterministic across processes.

## Composite-key bytes form

Mirror what ``TablespaceLongRegion`` already uses as its dict key
(``(model, compat_key, key)``) by canonicalizing it to bytes with NUL
separators:

    composite_bytes = model.encode() + b"\\x00" + compat_key.encode()
                    + b"\\x00" + key

Different (model, compat_key) tenants thus hash to different shards even
for the same raw key — and the same composite always picks the same
shard, restart-survival included.

## Why not share state between shards

Each shard is a complete ``TablespaceLongRegion`` (own journal, own
snapshot, own bitset allocator, own container files). No cross-shard
locking, no cross-shard journal coordination. A shard whose disk fails
takes itself offline (TablespaceLongRegion's existing flusher_failing
sticky-error) without dragging down the others.

## Hard constraints (see task spec)

- Each shard's read path remains ``os.pread`` / ``_aligned_read``.
- ``ThreadPoolExecutor`` is the only concurrency primitive — no asyncio.
- ``StripedLongRegion`` is a drop-in for ``TablespaceLongRegion``'s public
  surface (same put/get/exists/remove/clear signatures) so HostStore wires
  through unchanged.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from infera.kvd.tablespace import TablespaceEntry, TablespaceLongRegion

logger = logging.getLogger(__name__)


def composite_bytes(model: str, compat_key: str, key: bytes) -> bytes:
    """Canonical byte form of the (model, compat_key, key) composite
    used for shard routing. NUL separators avoid the collision where
    e.g. (model="abc", compat_key="def") and (model="abcd", compat_key="ef")
    would map to the same byte string if naively concatenated."""
    return model.encode("utf-8") + b"\x00" + compat_key.encode("utf-8") + b"\x00" + key


class StripedLongRegion:
    """Hash-keyed striping across N TablespaceLongRegion shards.

    For each composite (model, compat_key, key), picks shard
    ``hash(composite_bytes) % N``. All writes for a key go to the same
    shard; reads go to the same shard. Cross-shard operations (clear,
    multi-key exists, get_bytes_batch) fan out concurrently via a
    ThreadPoolExecutor sized to N.

    Construction:

        shards = [
            TablespaceLongRegion("/mnt/nvme0/kvd", max_bytes=4*1024**3, ...),
            TablespaceLongRegion("/mnt/nvme1/kvd", max_bytes=4*1024**3, ...),
            ...,
        ]
        region = StripedLongRegion(shards, workers_per_shard=4)
        region.start()

    ``workers_per_shard`` controls intra-shard parallelism: each shard's
    sub-batch in ``get_bytes_batch`` is further split across this many
    sub-workers so the per-shard pread storm pipelines. Default 8 is
    empirically the knee of the curve on 8-NVMe (see bench): 1→2→4 each
    add ~1 GB/s, 4→8 jumps from 6.3 to 9.6 GB/s, 8→16 only adds 0.5
    GB/s.

    A pre-built ``ThreadPoolExecutor`` may be passed; if not, one is
    created sized to ``len(shards) × workers_per_shard`` with daemon
    threads, and is shut down on ``shutdown()``.
    """

    def __init__(
        self,
        shards: list[TablespaceLongRegion],
        *,
        workers_per_shard: int = 8,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if not shards:
            raise ValueError("striped long region requires at least one shard")
        if workers_per_shard < 1:
            raise ValueError(f"workers_per_shard must be >= 1, got {workers_per_shard}")
        self._shards = list(shards)
        self._n = len(self._shards)
        self._workers_per_shard = int(workers_per_shard)
        total_workers = self._n * self._workers_per_shard
        # Default: workers_per_shard threads per shard, daemonized so
        # daemon teardown doesn't block on a stuck worker. The
        # "kvd-stripe" prefix makes the threads identifiable in
        # py-spy / perf flame graphs.
        self._executor = executor or ThreadPoolExecutor(
            max_workers=total_workers,
            thread_name_prefix="kvd-stripe",
        )
        self._owns_executor = executor is None
        self._started = False

    @property
    def workers_per_shard(self) -> int:
        return self._workers_per_shard

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start every shard in parallel. We submit one ``start()`` job
        per shard and gather — startup is dominated by fallocate +
        snapshot replay (sequential I/O per shard), so N shards on N
        devices want to fanout."""
        if self._started:
            return
        futures = [self._executor.submit(s.start) for s in self._shards]
        first_exc: BaseException | None = None
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                if first_exc is None:
                    first_exc = exc
                logger.exception("striped: shard start failed")
        if first_exc is not None:
            # Tear down any shards that DID start, then re-raise. We
            # don't want a partial striped region — keys hash-routed
            # to a dead shard would silently miss forever.
            for s in self._shards:
                try:
                    s.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            raise first_exc
        self._started = True

        logger.info(
            "striped long region started with %d shards: %s",
            self._n,
            [str(s.path) for s in self._shards],
        )

    def shutdown(self) -> None:
        if not self._started:
            return
        futures = [self._executor.submit(s.shutdown) for s in self._shards]
        for fut in futures:
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                logger.exception("striped: shard shutdown failed")
        self._started = False
        if self._owns_executor:
            # Don't wait — workers are daemonized and any leftover
            # shard call has already returned (we joined above).
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public properties — sum across shards
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "tablespace_striped"

    @property
    def max_bytes(self) -> int:
        return sum(s.max_bytes for s in self._shards)

    @property
    def used_bytes(self) -> int:
        return sum(s.used_bytes for s in self._shards)

    @property
    def entries_count(self) -> int:
        return sum(s.entries_count for s in self._shards)

    @property
    def num_shards(self) -> int:
        return self._n

    @property
    def shards(self) -> tuple[TablespaceLongRegion, ...]:
        return tuple(self._shards)

    @property
    def path(self) -> Path:
        """Returns the first shard's path. Provided for API parity with
        the underlying TablespaceLongRegion — StripedLongRegion doesn't
        have ONE root path, but HostStore's diagnostics still need a
        ``.path`` attribute to log."""
        return self._shards[0].path

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _pick(self, composite: bytes) -> int:
        """Shard index for a composite-bytes key.

        ``hashlib.blake2b`` with an 8-byte digest is the cheapest
        cryptographic-strength hash in stdlib (~250 ns/key on Zen 4,
        faster than sha256). Determinism across kvd restarts is the
        load-bearing property — Python's ``hash(bytes)`` is salted per
        process via PYTHONHASHSEED, so we'd lose every cached block
        on restart with that. blake2b also distributes much better
        than CRC32 / fnv at any key size we care about."""
        digest = hashlib.blake2b(composite, digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._n

    def _pick_key(self, key: bytes, model: str, compat_key: str) -> int:
        return self._pick(composite_bytes(model, compat_key, key))

    # ------------------------------------------------------------------
    # Single-key API — delegates to one shard, no executor
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
        idx = self._pick_key(key, model, compat_key)
        return self._shards[idx].put(
            key,
            value,
            retention=retention,
            model=model,
            compat_key=compat_key,
            metadata=metadata,
        )

    def get_bytes(self, key: bytes, *, model: str = "", compat_key: str = "") -> bytes | None:
        idx = self._pick_key(key, model, compat_key)
        return self._shards[idx].get_bytes(key, model=model, compat_key=compat_key)

    def get_entry(
        self, key: bytes, *, model: str = "", compat_key: str = ""
    ) -> TablespaceEntry | None:
        idx = self._pick_key(key, model, compat_key)
        return self._shards[idx].get_entry(key, model=model, compat_key=compat_key)

    def remove(self, key: bytes, *, model: str = "", compat_key: str = "") -> bool:
        idx = self._pick_key(key, model, compat_key)
        return self._shards[idx].remove(key, model=model, compat_key=compat_key)

    # ------------------------------------------------------------------
    # Multi-key API — fans out via executor
    # ------------------------------------------------------------------

    def exists(self, keys: list[bytes], *, model: str = "", compat_key: str = "") -> list[bool]:
        """Per-key membership. Each key goes to exactly one shard, so
        we can short-circuit by grouping keys per shard and only asking
        each shard about its keys."""
        if not keys:
            return []

        by_shard: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        for i, k in enumerate(keys):
            by_shard[self._pick_key(k, model, compat_key)].append((i, k))

        # Fanout: each shard does an in-memory dict lookup; the per-shard
        # cost is microseconds — but submitting via the executor still
        # helps when we're called from a hot path where multiple
        # exists/batch calls overlap.
        out = [False] * len(keys)

        def _check_shard(sid: int) -> list[bool]:
            items = by_shard[sid]
            sub_keys = [k for _, k in items]
            return self._shards[sid].exists(sub_keys, model=model, compat_key=compat_key)

        futures = {sid: self._executor.submit(_check_shard, sid) for sid in by_shard}
        for sid, fut in futures.items():
            sub_result = fut.result()
            for (orig_i, _k), present in zip(by_shard[sid], sub_result, strict=True):
                out[orig_i] = present
        return out

    def get_bytes_batch(
        self,
        keys: list[bytes],
        *,
        model: str = "",
        compat_key: str = "",
    ) -> list[bytes | None]:
        """Bulk read with two-level fanout.

        Level 1: keys are grouped by shard (hash-routed). Each shard's
        sub-batch is independent of the others — different inodes on
        different NVMe devices.

        Level 2: each shard's sub-batch is further chunked across
        ``workers_per_shard`` sub-workers. Each sub-worker calls the
        underlying shard's read path (``get_bytes_batch`` if present,
        otherwise a loop of ``get_bytes``). Since
        ``TablespaceLongRegion.get_bytes`` releases its dict lock
        before issuing pread, the sub-workers actually pipeline at the
        kernel level and the per-shard storm runs in parallel.

        With N=8 shards and workers_per_shard=4, this gives 32-way
        parallel pread fanout — empirically ~2-3× over the 1-worker-
        per-shard baseline on real NVMe (see benchmarks).

        Output preserves the caller's input order regardless of which
        sub-worker completed first.
        """
        if not keys:
            return []

        # Group keys by shard, preserving original index.
        by_shard: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        for i, k in enumerate(keys):
            by_shard[self._pick_key(k, model, compat_key)].append((i, k))

        # Split each shard's sub-batch into up to workers_per_shard
        # chunks and submit each chunk as its own future. ``futures``
        # carries the original (orig_i, key) tuples alongside the
        # future so we can re-stitch in input order without rehashing.
        futures: list[tuple[Any, list[tuple[int, bytes]]]] = []
        for sid, items in by_shard.items():
            n_subs = min(self._workers_per_shard, len(items))
            # Even chunking: ceil(len(items) / n_subs) per sub-batch.
            # The last chunk may be slightly smaller, which is fine —
            # imbalance of 1 key out of (e.g.) 16 is well below the
            # per-pread variance.
            chunk_size = (len(items) + n_subs - 1) // n_subs
            for ci in range(n_subs):
                chunk = items[ci * chunk_size : (ci + 1) * chunk_size]
                if not chunk:
                    # Defensive — uneven splits where workers_per_shard
                    # > len(items) means later chunks empty out. Skip
                    # them rather than submitting an empty future
                    # (would be wasted scheduler work).
                    continue
                sub_keys = [k for _, k in chunk]
                futures.append(
                    (
                        self._executor.submit(
                            _shard_batch_get,
                            self._shards[sid],
                            sub_keys,
                            model,
                            compat_key,
                        ),
                        chunk,
                    )
                )

        # Gather + re-order. Each future contributes results for its
        # own chunk's positions — the chunks across all shards
        # collectively cover every input position exactly once.
        out: list[bytes | None] = [None] * len(keys)
        for future, chunk in futures:
            sub_result = future.result()
            for (orig_i, _k), val in zip(chunk, sub_result, strict=True):
                out[orig_i] = val
        return out

    # ------------------------------------------------------------------
    # Cross-shard operations
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Clear every shard. Returns total entries dropped across
        shards. Fans out so an 8-device clear runs ~8× faster than
        sequential."""
        if not self._started:
            return 0
        futures = [self._executor.submit(s.clear) for s in self._shards]
        total = 0
        for fut in futures:
            try:
                total += fut.result()
            except Exception:  # noqa: BLE001
                logger.exception("striped: shard clear failed")
        return total

    def clear_namespace(self, model: str, compat_key: str) -> int:
        """Remove all entries from every shard matching (model,
        compat_key). Mirrors MultiPoolTablespaceLongRegion.clear_namespace
        for HostStore's _clear_ssd_namespace dispatch — without this,
        the host store reaches into a ``_entries`` attribute that
        doesn't exist on the wrapper and silently no-ops."""

        def _ns_clear(shard: TablespaceLongRegion) -> int:
            count = 0
            # Snapshot under the shard's own lock semantics — list()
            # on .items() is enough since TablespaceLongRegion guards
            # _entries with its own lock for mutation.
            entries_snapshot = list(shard._entries.items())
            for (m, ck, k), _ in entries_snapshot:
                if m == model and ck == compat_key:
                    if shard.remove(k, model=m, compat_key=ck):
                        count += 1
            return count

        futures = [self._executor.submit(_ns_clear, s) for s in self._shards]
        total = 0
        for fut in futures:
            total += fut.result()
        return total

    def stats(self) -> dict:
        """Aggregate per-shard counters into a single dict. Per-shard
        breakdown is included under ``shards`` for ops debugging
        (e.g. "is one device starving?")."""
        per_shard = []
        for i, s in enumerate(self._shards):
            per_shard.append(
                {
                    "shard_id": i,
                    "path": str(s.path),
                    "max_bytes": s.max_bytes,
                    "used_bytes": s.used_bytes,
                    "entries_count": s.entries_count,
                    "slot_bytes": s.slot_bytes,
                    "num_containers": s.num_containers,
                }
            )
        return {
            "num_shards": self._n,
            "entries_total": sum(s.entries_count for s in self._shards),
            "bytes_used_total": sum(s.used_bytes for s in self._shards),
            "bytes_max_total": sum(s.max_bytes for s in self._shards),
            "shards": per_shard,
        }


# ----------------------------------------------------------------------
# Per-shard batch get helper
# ----------------------------------------------------------------------


def _shard_batch_get(
    shard: TablespaceLongRegion,
    keys: list[bytes],
    model: str,
    compat_key: str,
) -> list[bytes | None]:
    """Use the shard's ``get_bytes_batch`` if present, otherwise fall
    back to a plain Python loop of ``get_bytes``. The read path is
    ``os.pread`` / ``_aligned_read`` either way."""
    batched = getattr(shard, "get_bytes_batch", None)
    if callable(batched):
        return batched(keys, model=model, compat_key=compat_key)
    return [shard.get_bytes(k, model=model, compat_key=compat_key) for k in keys]
