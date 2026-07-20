###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Snapshot reconciliation: self-healing for event drops.

Two pieces:

  - **SnapshotProducer**: worker / pool-daemon-side. Maintains a mirror of
    the events the publisher has emitted. Serves snapshots when the
    server asks. (The SGLang adapter can later replace this with a
    cheaper engine-direct walk; the interface is the same.)

  - **SnapshotReconciler**: server-side. Periodically (and on-demand
    after detected batch_id gaps) pulls each publisher's snapshot,
    acquires the per-tree write lock from the KvIndexWriter, drops the
    tree, and re-applies the snapshot's blocks.

The two communicate via JSON over HTTP at low frequency (~30 s default).
At high frequency the ZMQ event stream is authoritative; snapshots only
recover from drift.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from infera.kv.index import KVIndex
from infera.kv.types import BlockKey, Tier
from infera.kv.wire import (
    EventType,
    NormalizedEvent,
    Snapshot,
    SnapshotBlock,
)
from infera.kv.writer import KvIndexWriter

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Producer side (worker / pool daemon)
# ----------------------------------------------------------------------


class SnapshotProducer:
    """In-memory mirror of what the publisher has emitted.

    Probes call `on_event(model, compat_key, event)` for every event they
    publish via ZMQ. The producer accumulates the same state in a small
    map so it can answer `GET /v1/kv-snapshot` without consulting the
    engine.

    Trade-off: doubles the memory of the publisher's cache footprint at
    block-metadata granularity (~48 B per block). For a worker with
    100k cached blocks that's ~5 MB — acceptable.

    Engine-direct alternative: SGLang's RadixCache could be walked
    directly to produce a snapshot; that avoids the mirror but adds
    engine-specific code. v1 ships the mirror.
    """

    def __init__(
        self,
        *,
        publisher_id: str,
        publisher_type: str,
        index_block_size: int,
    ) -> None:
        self._publisher_id = publisher_id
        self._publisher_type = publisher_type
        self._index_block_size = index_block_size
        # (model, compat_key) → {sequence_hash → SnapshotBlock}
        self._state: dict[tuple[str, str], dict[int, SnapshotBlock]] = {}
        # (model, compat_key) → highest batch_id seen
        self._batch_ids: dict[tuple[str, str], int] = {}

    def on_event(
        self,
        *,
        model: str,
        compat_key: str,
        event: NormalizedEvent,
        batch_id: int | None = None,
    ) -> None:
        """Mirror one event into the producer's state. Called by the probe
        for every event it emits to the ZMQ publisher.

        `batch_id` should be the batch_id the publisher will assign to
        the batch that contains this event. The reconciler uses the
        snapshot's reported batch_id to know the freshness watermark.
        """
        key = (model, compat_key)
        if batch_id is not None and batch_id > self._batch_ids.get(key, -1):
            self._batch_ids[key] = batch_id

        if event.type == EventType.STORED:
            if event.sequence_hash is None or event.block_hash is None or event.tier is None:
                return
            blocks = self._state.setdefault(key, {})
            existing = blocks.get(event.sequence_hash)
            if existing is None:
                blocks[event.sequence_hash] = SnapshotBlock(
                    sequence_hash=event.sequence_hash,
                    parent_sequence_hash=event.parent_sequence_hash,
                    block_hash=event.block_hash,
                    tiers=(event.tier,),
                )
            else:
                if event.tier not in existing.tiers:
                    blocks[event.sequence_hash] = SnapshotBlock(
                        sequence_hash=existing.sequence_hash,
                        parent_sequence_hash=existing.parent_sequence_hash,
                        block_hash=existing.block_hash,
                        tiers=existing.tiers + (event.tier,),
                    )
        elif event.type == EventType.REMOVED:
            if event.sequence_hash is None or event.tier is None:
                return
            blocks = self._state.get(key)
            if blocks is None:
                return
            existing = blocks.get(event.sequence_hash)
            if existing is None:
                return
            remaining = tuple(t for t in existing.tiers if t != event.tier)
            if not remaining:
                blocks.pop(event.sequence_hash, None)
            else:
                blocks[event.sequence_hash] = SnapshotBlock(
                    sequence_hash=existing.sequence_hash,
                    parent_sequence_hash=existing.parent_sequence_hash,
                    block_hash=existing.block_hash,
                    tiers=remaining,
                )
        elif event.type == EventType.CLEARED:
            self._apply_cleared(event.scope or "all")

    def _apply_cleared(self, scope: str) -> None:
        if scope == "all":
            self._state.clear()
            self._batch_ids.clear()
            return
        if scope.startswith("model:"):
            target = scope.removeprefix("model:")
            for k in [k for k in self._state if k[0] == target]:
                self._state.pop(k, None)
                self._batch_ids.pop(k, None)
            return
        if scope.startswith("compat_key:"):
            target = scope.removeprefix("compat_key:")
            for k in [k for k in self._state if k[1] == target]:
                self._state.pop(k, None)
                self._batch_ids.pop(k, None)
            return
        if scope.startswith("tier:"):
            target = scope.removeprefix("tier:")
            for blocks in self._state.values():
                for seq, blk in list(blocks.items()):
                    remaining = tuple(t for t in blk.tiers if t != target)
                    if not remaining:
                        blocks.pop(seq, None)
                    else:
                        blocks[seq] = SnapshotBlock(
                            sequence_hash=blk.sequence_hash,
                            parent_sequence_hash=blk.parent_sequence_hash,
                            block_hash=blk.block_hash,
                            tiers=remaining,
                        )

    def keys(self) -> list[tuple[str, str]]:
        """List (model, compat_key) pairs this producer can snapshot."""
        return list(self._state.keys())

    def snapshot(self, *, model: str, compat_key: str) -> Snapshot:
        """Produce a snapshot for one (model, compat_key). If no events
        have been emitted yet, returns an empty snapshot with batch_id=-1
        (the reconciler ignores empty snapshots until they have content).
        """
        key = (model, compat_key)
        blocks = self._state.get(key, {})
        return Snapshot(
            publisher_id=self._publisher_id,
            publisher_type=self._publisher_type,
            model_name=model,
            compat_key=compat_key,
            index_block_size=self._index_block_size,
            batch_id=self._batch_ids.get(key, -1),
            blocks=tuple(blocks.values()),
        )


# ----------------------------------------------------------------------
# Reconciler side (server)
# ----------------------------------------------------------------------


PullFn = Callable[[str, str, str, str], Awaitable[Snapshot | None]]
# (publisher_id, endpoint, model, compat_key) -> Snapshot or None on transient failure


class SnapshotReconciler:
    """Pulls snapshots from publishers and rebuilds the per-tree state."""

    def __init__(
        self,
        *,
        index: KVIndex,
        writer: KvIndexWriter,
        pull_fn: PullFn,
        interval_s: float = 30.0,
    ) -> None:
        self._index = index
        self._writer = writer
        self._pull_fn = pull_fn
        self._interval_s = interval_s

        # (publisher_id, endpoint, model, compat_key) -> any pending state
        self._targets: dict[tuple[str, str, str, str], None] = {}
        # (publisher_id, endpoint, model, compat_key) -> last successful batch_id
        self._last_seen_batch: dict[tuple[str, str, str, str], int] = {}
        # On-demand kick: set this to wake the periodic loop immediately.
        self._kick = asyncio.Event()
        self._urgent: set[tuple[str, str, str, str]] = set()
        self._task: asyncio.Task | None = None
        self._started = False
        self._closing = False

        # Metrics
        self.snapshots_pulled = 0
        self.snapshots_applied = 0
        self.snapshots_failed = 0
        self.snapshots_stale = 0

    def register_target(
        self,
        *,
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
    ) -> None:
        """Tell the reconciler to periodically pull this (publisher, tree)."""
        key = (publisher_id, endpoint, model, compat_key)
        self._targets[key] = None

    def unregister_target(
        self,
        *,
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
    ) -> None:
        key = (publisher_id, endpoint, model, compat_key)
        self._targets.pop(key, None)
        self._last_seen_batch.pop(key, None)
        self._urgent.discard(key)

    def trigger_gap_recovery(
        self,
        *,
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
    ) -> None:
        """Subscriber detected a batch_id gap from this publisher → pull
        the snapshot at the next loop iteration without waiting for the
        timer. Idempotent; coalesces multiple kicks into one pull.
        """
        key = (publisher_id, endpoint, model, compat_key)
        if key in self._targets:
            self._urgent.add(key)
            self._kick.set()

    async def start(self) -> None:
        if self._started:
            return
        self._task = asyncio.create_task(self._loop(), name="kv-snapshot-reconciler")
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._closing = True
        self._kick.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._started = False

    async def _loop(self) -> None:
        try:
            while not self._closing:
                # Run urgent first.
                if self._urgent:
                    pending = list(self._urgent)
                    self._urgent.clear()
                    for key in pending:
                        await self._reconcile_one(key)
                # Then the periodic sweep.
                for key in list(self._targets):
                    await self._reconcile_one(key)
                # Wait until the next periodic tick or a kick.
                try:
                    await asyncio.wait_for(self._kick.wait(), timeout=self._interval_s)
                except asyncio.TimeoutError:
                    pass
                self._kick.clear()
        except asyncio.CancelledError:
            return

    async def reconcile_now(
        self,
        *,
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
    ) -> bool:
        """Trigger an immediate reconcile for one target. Returns True
        if the snapshot was successfully applied. Useful for tests and
        for explicit operator invalidation flow."""
        key = (publisher_id, endpoint, model, compat_key)
        return await self._reconcile_one(key)

    async def _reconcile_one(self, key: tuple[str, str, str, str]) -> bool:
        publisher_id, endpoint, model, compat_key = key
        try:
            snapshot = await self._pull_fn(publisher_id, endpoint, model, compat_key)
        except Exception as exc:
            self.snapshots_failed += 1
            logger.warning(
                "snapshot pull failed for %s @ %s (%s/%s): %s",
                publisher_id,
                endpoint,
                model,
                compat_key,
                exc,
            )
            return False
        if snapshot is None:
            self.snapshots_failed += 1
            return False
        self.snapshots_pulled += 1

        # Don't bother applying if we've already seen a newer event stream
        # for this tree. A snapshot whose batch_id is older than what we've
        # already digested would be a regression.
        last_seen = self._last_seen_batch.get(key, -1)
        if snapshot.batch_id != -1 and snapshot.batch_id < last_seen:
            self.snapshots_stale += 1
            logger.debug(
                "snapshot stale for %s (snapshot.batch_id=%d, last_seen=%d); skipping",
                key,
                snapshot.batch_id,
                last_seen,
            )
            return False

        # Acquire the per-tree write lock so concurrent event apply doesn't
        # race with the rebuild.
        lock = self._writer.lock_for(model, compat_key, publisher_id)
        async with lock:
            self._index.drop_tree(model=model, compat_key=compat_key, publisher_id=publisher_id)
            for blk in snapshot.blocks:
                if not blk.tiers:
                    continue
                block_key = BlockKey(
                    sequence_hash=blk.sequence_hash,
                    block_hash=blk.block_hash,
                    parent_sequence_hash=blk.parent_sequence_hash,
                )
                for tier_str in blk.tiers:
                    try:
                        tier = Tier(tier_str)
                    except ValueError:
                        continue
                    self._index.apply_stored(
                        publisher_id=publisher_id,
                        model=model,
                        compat_key=compat_key,
                        block=block_key,
                        tier=tier,
                    )
        if snapshot.batch_id != -1:
            self._last_seen_batch[key] = snapshot.batch_id
        self.snapshots_applied += 1
        return True

    def note_batch_seen(
        self,
        *,
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
        batch_id: int,
    ) -> None:
        """Called by the writer (or its caller) when a fresh batch is
        applied, so the reconciler knows what's already in the index.
        Prevents an old snapshot from overwriting newer event-applied
        state."""
        key = (publisher_id, endpoint, model, compat_key)
        if batch_id > self._last_seen_batch.get(key, -1):
            self._last_seen_batch[key] = batch_id

    def note_batch_seen_by_stream(
        self,
        *,
        publisher_id: str,
        model: str,
        compat_key: str,
        batch_id: int,
    ) -> None:
        """Same as `note_batch_seen` but the caller doesn't need to know
        the snapshot endpoint. Updates every registered target matching
        (publisher_id, model, compat_key) — endpoint is informational
        for the puller, not part of the dedup decision.

        This is the path the `KvIndexWriter` calls after each event
        batch lands. Without it the staleness check is permanently
        disarmed and every 30 s reconcile wipes + partially rebuilds
        the per-tree index from a snapshot older than what's already
        been applied.
        """
        for key in self._targets:
            k_pub, _endpoint, k_model, k_compat = key
            if k_pub == publisher_id and k_model == model and k_compat == compat_key:
                if batch_id > self._last_seen_batch.get(key, -1):
                    self._last_seen_batch[key] = batch_id
