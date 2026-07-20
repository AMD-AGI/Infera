###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Single-writer task that drains the event queue and applies batches to KVIndex.

One task per server replica. Holds a per-(model, compat_key, publisher_id)
`asyncio.Lock` while applying a batch — so writes to one tree don't block
reads on another.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from infera.kv.index import IndexKey, KVIndex
from infera.kv.types import BlockKey, Tier
from infera.kv.wire import EventBatch, EventType

if TYPE_CHECKING:
    # Avoid runtime circular import: snapshot.py imports writer.py.
    from infera.kv.snapshot import SnapshotReconciler

logger = logging.getLogger(__name__)


class WriterMetrics:
    def __init__(self) -> None:
        self.batches_applied = 0
        self.events_applied_stored = 0
        self.events_applied_removed = 0
        self.events_applied_cleared = 0
        self.events_dropped_unknown_role = 0
        self.events_dropped_unknown_tier = 0
        self.events_dropped_malformed = 0


class KvIndexWriter:
    """Async task that pulls EventBatch from a queue and applies to KVIndex.

    The KVIndex itself has synchronous methods (no locks inside). This
    writer is the only mutation source, and it holds per-tree
    `asyncio.Lock`s so concurrent readers (the routing policy) can grab
    the same lock before walking the tree.

    Locks are exposed via `lock_for(...)` so reader paths can use the
    same instance as the writer.
    """

    def __init__(self, *, index: KVIndex, queue: asyncio.Queue) -> None:
        self._index = index
        self._queue = queue
        self._locks: dict[IndexKey, asyncio.Lock] = {}
        self._task: asyncio.Task | None = None
        self._started = False
        self._closing = False
        self.metrics = WriterMetrics()
        # Optional reconciler — set after construction via `set_reconciler`
        # because reconciler takes a writer ref (circular dep). When set,
        # `_apply_batch` informs it of each applied batch_id so the
        # staleness check correctly skips snapshots older than already-
        # applied events. Without this wiring, the 30 s reconcile tick
        # wipes + partially rebuilds the index from stale snapshots.
        # Optional so unit tests that don't construct a reconciler can
        # still drive the writer in isolation.
        self._reconciler: SnapshotReconciler | None = None

    def set_reconciler(self, reconciler: SnapshotReconciler) -> None:
        """Attach the snapshot reconciler. Called after construction by
        the server's main loop, breaks the writer ↔ reconciler circular
        construction dependency."""
        self._reconciler = reconciler

    def lock_for(self, model: str, compat_key: str, publisher_id: str) -> asyncio.Lock:
        """Return the lock guarding a specific (model, compat_key, publisher_id)
        tree. Readers should acquire this before calling KVIndex.find_matches
        for that tree.
        """
        key: IndexKey = (model, compat_key, publisher_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def start(self) -> None:
        if self._started:
            return
        self._task = asyncio.create_task(self._writer_loop(), name="kv-writer")
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._started = False

    async def _writer_loop(self) -> None:
        try:
            while not self._closing:
                batch = await self._queue.get()
                await self._apply_batch(batch)
        except asyncio.CancelledError:
            return

    async def _apply_batch(self, batch: EventBatch) -> None:
        # Cleared events scope outside one (model, compat_key) — e.g. "all"
        # spans every tree this publisher owns — so we acquire the lock per
        # event for cleared, and per-batch for stored/removed.
        # In practice batches contain a single type, so this is fine.
        for event in batch.events:
            try:
                if event.type == EventType.CLEARED:
                    # apply_cleared with scope="all" / "compat_key:*" /
                    # "tier:*" can mutate ANY tree the publisher owns,
                    # not just the batch's tree. Holding only the
                    # batch's lock leaves readers walking OTHER trees
                    # for the same publisher racing the drop (PR #9
                    # review fix P1). Acquire every relevant tree's
                    # lock before applying.
                    publisher_id = batch.publisher_id
                    scope = event.scope or "all"
                    # Collect all locks for this publisher's trees.
                    # Snapshot the index keys via the private `_trees`
                    # dict — safe under the GIL since we only iterate.
                    locks_to_take = [
                        self.lock_for(m, ck, pub)
                        for (m, ck, pub) in list(self._index._trees.keys())
                        if pub == publisher_id
                    ]
                    # Always include the batch's lock — covers the
                    # case where the cleared event arrives before any
                    # STORED events for that tree.
                    batch_lock = self.lock_for(batch.model_name, batch.compat_key, publisher_id)
                    if batch_lock not in locks_to_take:
                        locks_to_take.append(batch_lock)
                    # Acquire in stable order (by id) to avoid the
                    # theoretical multi-writer deadlock. Writer is
                    # single-task today, but stable ordering keeps
                    # this safe under future fan-out.
                    locks_to_take.sort(key=id)
                    async with contextlib.AsyncExitStack() as stack:
                        for lock in locks_to_take:
                            await stack.enter_async_context(lock)
                        self._index.apply_cleared(
                            publisher_id=publisher_id,
                            scope=scope,
                        )
                    self.metrics.events_applied_cleared += 1
                elif event.type == EventType.STORED:
                    if (
                        event.tier is None
                        or event.sequence_hash is None
                        or event.block_hash is None
                    ):
                        self.metrics.events_dropped_malformed += 1
                        continue
                    try:
                        tier = Tier(event.tier)
                    except ValueError:
                        self.metrics.events_dropped_unknown_tier += 1
                        continue
                    block = BlockKey(
                        sequence_hash=event.sequence_hash,
                        block_hash=event.block_hash,
                        parent_sequence_hash=event.parent_sequence_hash,
                    )
                    lock = self.lock_for(batch.model_name, batch.compat_key, batch.publisher_id)
                    async with lock:
                        self._index.apply_stored(
                            publisher_id=batch.publisher_id,
                            model=batch.model_name,
                            compat_key=batch.compat_key,
                            block=block,
                            tier=tier,
                        )
                    self.metrics.events_applied_stored += 1
                elif event.type == EventType.REMOVED:
                    if event.tier is None or event.sequence_hash is None:
                        self.metrics.events_dropped_malformed += 1
                        continue
                    try:
                        tier = Tier(event.tier)
                    except ValueError:
                        self.metrics.events_dropped_unknown_tier += 1
                        continue
                    lock = self.lock_for(batch.model_name, batch.compat_key, batch.publisher_id)
                    async with lock:
                        self._index.apply_removed(
                            publisher_id=batch.publisher_id,
                            model=batch.model_name,
                            compat_key=batch.compat_key,
                            sequence_hash=event.sequence_hash,
                            tier=tier,
                        )
                    self.metrics.events_applied_removed += 1
                else:
                    self.metrics.events_dropped_malformed += 1
            except Exception:
                # Don't let one bad event kill the writer.
                logger.exception("failed to apply event: %s", event)
                self.metrics.events_dropped_malformed += 1
        self.metrics.batches_applied += 1
        # Tell the reconciler what's been applied so its periodic 30 s
        # tick doesn't overwrite this with a stale snapshot. The
        # `_last_seen_batch[(publisher, *, model, compat_key)]` map
        # gates the staleness check in `_reconcile_one`. Closes the
        # P0 "missing note_batch_seen wiring" finding from the PR #9
        # review.
        if self._reconciler is not None:
            try:
                self._reconciler.note_batch_seen_by_stream(
                    publisher_id=batch.publisher_id,
                    model=batch.model_name,
                    compat_key=batch.compat_key,
                    batch_id=batch.batch_id,
                )
            except Exception:
                # Best-effort — staleness protection failure isn't worth
                # killing the writer. Logged for ops visibility.
                logger.exception(
                    "note_batch_seen_by_stream failed (publisher=%s batch=%d)",
                    batch.publisher_id,
                    batch.batch_id,
                )
