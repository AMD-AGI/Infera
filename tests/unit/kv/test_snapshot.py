###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio

import pytest

from infera.kv.hashing import hash_token_blocks
from infera.kv.index import KVIndex
from infera.kv.snapshot import SnapshotProducer, SnapshotReconciler
from infera.kv.types import OverlapBlocks, Tier
from infera.kv.wire import (
    EVENT_VERSION,
    EventBatch,
    Snapshot,
    SnapshotBlock,
    make_cleared_event,
    make_removed_event,
    make_stored_event,
    snapshot_from_json,
    snapshot_to_json,
)
from infera.kv.writer import KvIndexWriter

# ----------------------------------------------------------------------
# Snapshot JSON wire format
# ----------------------------------------------------------------------


def test_snapshot_json_round_trip() -> None:
    snap = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="Qwen3.6",
        compat_key="abcdef0123456789",
        index_block_size=64,
        batch_id=42,
        blocks=(
            SnapshotBlock(
                sequence_hash=0x111,
                parent_sequence_hash=None,
                block_hash=0x222,
                tiers=("device", "host"),
            ),
            SnapshotBlock(
                sequence_hash=0x333,
                parent_sequence_hash=0x111,
                block_hash=0x444,
                tiers=("device",),
            ),
        ),
    )
    obj = snapshot_to_json(snap)
    # Large integers are encoded as 0x-prefixed hex.
    assert obj["blocks"][0]["sequence_hash"].startswith("0x")
    decoded = snapshot_from_json(obj)
    assert decoded == snap


def test_snapshot_json_accepts_int_or_hex() -> None:
    obj = {
        "v": EVENT_VERSION,
        "publisher_id": "w1",
        "publisher_type": "worker",
        "model_name": "m",
        "compat_key": "ck",
        "index_block_size": 64,
        "batch_id": 0,
        "blocks": [
            {
                "sequence_hash": 100,  # int form
                "parent_sequence_hash": None,
                "block_hash": "0x200",  # hex form
                "tiers": ["device"],
            }
        ],
    }
    snap = snapshot_from_json(obj)
    assert snap.blocks[0].sequence_hash == 100
    assert snap.blocks[0].block_hash == 0x200


def test_snapshot_json_rejects_unknown_version() -> None:
    obj = {
        "v": EVENT_VERSION + 99,
        "publisher_id": "w1",
        "publisher_type": "worker",
        "model_name": "m",
        "compat_key": "ck",
        "index_block_size": 64,
        "batch_id": 0,
        "blocks": [],
    }
    with pytest.raises(ValueError, match="newer than supported"):
        snapshot_from_json(obj)


# ----------------------------------------------------------------------
# SnapshotProducer (worker-side mirror)
# ----------------------------------------------------------------------


def _producer() -> SnapshotProducer:
    return SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)


def test_producer_records_stored_event() -> None:
    p = _producer()
    chain = hash_token_blocks(list(range(8)), block_size=4)
    for _i, block in enumerate(chain):
        p.on_event(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=block.sequence_hash,
                block_hash=block.block_hash,
                parent_sequence_hash=block.parent_sequence_hash,
                tier="device",
            ),
            batch_id=0,
        )
    snap = p.snapshot(model="m", compat_key="ck")
    assert len(snap.blocks) == 2
    assert snap.blocks[0].tiers == ("device",)
    assert snap.batch_id == 0


def test_producer_multi_tier_merges() -> None:
    p = _producer()
    chain = hash_token_blocks(list(range(4)), block_size=4)
    for tier in ("device", "host"):
        p.on_event(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=chain[0].sequence_hash,
                block_hash=chain[0].block_hash,
                parent_sequence_hash=chain[0].parent_sequence_hash,
                tier=tier,
            ),
            batch_id=0,
        )
    snap = p.snapshot(model="m", compat_key="ck")
    assert set(snap.blocks[0].tiers) == {"device", "host"}


def test_producer_removed_drops_tier_then_block() -> None:
    p = _producer()
    chain = hash_token_blocks(list(range(4)), block_size=4)
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_stored_event(
            sequence_hash=chain[0].sequence_hash,
            block_hash=chain[0].block_hash,
            parent_sequence_hash=chain[0].parent_sequence_hash,
            tier="device",
        ),
    )
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_stored_event(
            sequence_hash=chain[0].sequence_hash,
            block_hash=chain[0].block_hash,
            parent_sequence_hash=chain[0].parent_sequence_hash,
            tier="host",
        ),
    )
    # Remove device — host remains.
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_removed_event(sequence_hash=chain[0].sequence_hash, tier="device"),
    )
    snap = p.snapshot(model="m", compat_key="ck")
    assert snap.blocks[0].tiers == ("host",)
    # Remove host — block fully gone.
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_removed_event(sequence_hash=chain[0].sequence_hash, tier="host"),
    )
    snap = p.snapshot(model="m", compat_key="ck")
    assert snap.blocks == ()


def test_producer_cleared_scopes() -> None:
    p = _producer()
    chain = hash_token_blocks(list(range(4)), block_size=4)
    for model in ("m1", "m2"):
        for ck in ("ckA", "ckB"):
            p.on_event(
                model=model,
                compat_key=ck,
                event=make_stored_event(
                    sequence_hash=chain[0].sequence_hash,
                    block_hash=chain[0].block_hash,
                    parent_sequence_hash=chain[0].parent_sequence_hash,
                    tier="device",
                ),
            )
    # Clear one model.
    p.on_event(
        model="m1",
        compat_key="ckA",
        event=make_cleared_event(scope="model:m1"),
    )
    assert p.snapshot(model="m1", compat_key="ckA").blocks == ()
    assert p.snapshot(model="m1", compat_key="ckB").blocks == ()
    assert len(p.snapshot(model="m2", compat_key="ckA").blocks) == 1

    # Clear a compat_key.
    p.on_event(
        model="m2",
        compat_key="ckA",
        event=make_cleared_event(scope="compat_key:ckA"),
    )
    assert p.snapshot(model="m2", compat_key="ckA").blocks == ()
    assert len(p.snapshot(model="m2", compat_key="ckB").blocks) == 1

    # Clear all.
    p.on_event(
        model="m2",
        compat_key="ckB",
        event=make_cleared_event(scope="all"),
    )
    assert p.snapshot(model="m2", compat_key="ckB").blocks == ()


def test_producer_batch_id_tracked_per_stream() -> None:
    p = _producer()
    chain = hash_token_blocks(list(range(4)), block_size=4)
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_stored_event(
            sequence_hash=chain[0].sequence_hash,
            block_hash=chain[0].block_hash,
            parent_sequence_hash=chain[0].parent_sequence_hash,
            tier="device",
        ),
        batch_id=10,
    )
    # Newer batch_id overwrites.
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_stored_event(
            sequence_hash=chain[0].sequence_hash,
            block_hash=chain[0].block_hash,
            parent_sequence_hash=chain[0].parent_sequence_hash,
            tier="host",
        ),
        batch_id=15,
    )
    snap = p.snapshot(model="m", compat_key="ck")
    assert snap.batch_id == 15

    # Older batch_id does NOT overwrite.
    p.on_event(
        model="m",
        compat_key="ck",
        event=make_stored_event(
            sequence_hash=chain[0].sequence_hash,
            block_hash=chain[0].block_hash,
            parent_sequence_hash=chain[0].parent_sequence_hash,
            tier="disk",
        ),
        batch_id=5,
    )
    snap = p.snapshot(model="m", compat_key="ck")
    assert snap.batch_id == 15  # still 15, not 5


def test_producer_empty_snapshot_for_unknown_stream() -> None:
    p = _producer()
    snap = p.snapshot(model="never-seen", compat_key="anything")
    assert snap.blocks == ()
    assert snap.batch_id == -1


# ----------------------------------------------------------------------
# SnapshotReconciler
# ----------------------------------------------------------------------


async def _make_reconciler(pull_fn) -> tuple[KVIndex, KvIndexWriter, SnapshotReconciler]:
    index = KVIndex()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    writer = KvIndexWriter(index=index, queue=queue)
    await writer.start()
    rec = SnapshotReconciler(index=index, writer=writer, pull_fn=pull_fn, interval_s=10_000)
    return index, writer, rec


async def test_reconciler_applies_snapshot_to_empty_index() -> None:
    chain = hash_token_blocks(list(range(8)), block_size=4)
    snap = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=0,
        blocks=tuple(
            SnapshotBlock(
                sequence_hash=b.sequence_hash,
                parent_sequence_hash=b.parent_sequence_hash,
                block_hash=b.block_hash,
                tiers=("device",),
            )
            for b in chain
        ),
    )

    async def pull_fn(_pub, _ep, _model, _ck):
        return snap

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        ok = await rec.reconcile_now(
            publisher_id="w1", endpoint="ignored", model="m", compat_key="ck"
        )
        assert ok is True
        # Index has the snapshot's content.
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks(device=2)
        assert rec.snapshots_applied == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_reconciler_overwrites_existing_state() -> None:
    """An existing tree is dropped and replaced by the snapshot's content.
    Stale entries that should no longer be there get removed.
    """
    chain = hash_token_blocks(list(range(8)), block_size=4)
    # Snapshot only covers block 0 (engine evicted block 1, server missed event).
    snap = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=99,
        blocks=(
            SnapshotBlock(
                sequence_hash=chain[0].sequence_hash,
                parent_sequence_hash=None,
                block_hash=chain[0].block_hash,
                tiers=("device",),
            ),
        ),
    )

    async def pull_fn(*args):
        return snap

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        # Pre-populate index with both blocks (stale state).
        for block in chain:
            index.apply_stored(
                publisher_id="w1",
                model="m",
                compat_key="ck",
                block=block,
                tier=Tier.DEVICE,
            )
        # Sanity: both blocks present before reconcile.
        pre = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert pre["w1"].device == 2

        await rec.reconcile_now(publisher_id="w1", endpoint="ignored", model="m", compat_key="ck")

        # After reconcile, only block 0 remains.
        post = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert post["w1"].device == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_reconciler_skips_stale_snapshot() -> None:
    """If we've seen a batch_id higher than what the snapshot reports,
    skip it (don't regress newer event-stream state)."""
    chain = hash_token_blocks(list(range(4)), block_size=4)
    snap_old = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=5,
        blocks=(),
    )

    async def pull_fn(*args):
        return snap_old

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        # Pretend we've already seen batch_id 10 via the event stream.
        rec.note_batch_seen(
            publisher_id="w1",
            endpoint="ignored",
            model="m",
            compat_key="ck",
            batch_id=10,
        )
        # Pre-populate index with one block.
        index.apply_stored(
            publisher_id="w1",
            model="m",
            compat_key="ck",
            block=chain[0],
            tier=Tier.DEVICE,
        )

        ok = await rec.reconcile_now(
            publisher_id="w1", endpoint="ignored", model="m", compat_key="ck"
        )
        assert ok is False
        assert rec.snapshots_stale == 1
        # Index unchanged.
        matches = index.find_matches(
            model="m", compat_key="ck", chain=[chain[0]], candidates=["w1"]
        )
        assert matches["w1"].device == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_writer_feeds_reconciler_batch_seen_end_to_end() -> None:
    """Regression: the writer must
    inform the reconciler of each applied batch so a periodic
    reconcile doesn't overwrite newer state with an older snapshot.

    The pre-existing `test_reconciler_skips_stale_snapshot` calls
    `rec.note_batch_seen` directly — masks the wiring gap. This test
    exercises the REAL path: push a batch through the writer's queue,
    let it apply, then check the reconciler's staleness map.
    """
    # Pull function returns a stale snapshot at batch_id=5.
    chain = hash_token_blocks(list(range(8)), block_size=4)
    stale_snap = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=5,
        blocks=(),
    )

    async def pull_fn(*args):
        return stale_snap

    index, writer, rec = await _make_reconciler(pull_fn)
    # Make writer aware of reconciler — this is the wire that was missing.
    writer.set_reconciler(rec)
    # Register the stream so `note_batch_seen_by_stream` can update it.
    rec.register_target(publisher_id="w1", endpoint="http://w1/snap", model="m", compat_key="ck")
    try:
        # Push a batch at batch_id=10 through the queue (newer than snap).
        batch = EventBatch(
            publisher_id="w1",
            publisher_type="worker",
            model_name="m",
            compat_key="ck",
            index_block_size=4,
            batch_id=10,
            events=(
                make_stored_event(
                    sequence_hash=chain[0].sequence_hash,
                    block_hash=chain[0].block_hash,
                    parent_sequence_hash=chain[0].parent_sequence_hash,
                    tier="device",
                ),
            ),
        )
        await writer._queue.put(batch)

        # Wait for the writer to drain it.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if writer.metrics.batches_applied >= 1:
                break
        assert writer.metrics.batches_applied == 1

        # Reconciler should know about batch_id=10 now (via the wiring,
        # NOT via a direct note_batch_seen call from the test).
        key = ("w1", "http://w1/snap", "m", "ck")
        assert rec._last_seen_batch.get(key) == 10

        # Now try to apply the stale snapshot (batch_id=5). It should
        # be rejected because the reconciler knows we've already seen 10.
        ok = await rec.reconcile_now(
            publisher_id="w1", endpoint="http://w1/snap", model="m", compat_key="ck"
        )
        assert ok is False
        assert rec.snapshots_stale == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_writer_without_reconciler_still_works() -> None:
    """The reconciler reference is optional — unit tests that don't
    construct one should still drive the writer cleanly."""
    index = KVIndex()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    writer = KvIndexWriter(index=index, queue=queue)
    # Don't call set_reconciler.
    await writer.start()
    try:
        chain = hash_token_blocks(list(range(4)), block_size=4)
        await queue.put(
            EventBatch(
                publisher_id="w1",
                publisher_type="worker",
                model_name="m",
                compat_key="ck",
                index_block_size=4,
                batch_id=1,
                events=(
                    make_stored_event(
                        sequence_hash=chain[0].sequence_hash,
                        block_hash=chain[0].block_hash,
                        parent_sequence_hash=chain[0].parent_sequence_hash,
                        tier="device",
                    ),
                ),
            )
        )
        for _ in range(50):
            await asyncio.sleep(0.01)
            if writer.metrics.batches_applied >= 1:
                break
        assert writer.metrics.batches_applied == 1
    finally:
        await writer.stop()


async def test_reconciler_failed_pull_counts() -> None:
    async def pull_fn(*args):
        raise RuntimeError("network down")

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        ok = await rec.reconcile_now(
            publisher_id="w1", endpoint="ignored", model="m", compat_key="ck"
        )
        assert ok is False
        assert rec.snapshots_failed == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_reconciler_pull_returns_none_counts() -> None:
    async def pull_fn(*args):
        return None

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        ok = await rec.reconcile_now(
            publisher_id="w1", endpoint="ignored", model="m", compat_key="ck"
        )
        assert ok is False
        assert rec.snapshots_failed == 1
    finally:
        await rec.stop()
        await writer.stop()


async def test_reconciler_trigger_gap_recovery_runs_immediately() -> None:
    """Gap-detected by subscriber → reconciler runs that target soon."""
    chain = hash_token_blocks(list(range(4)), block_size=4)
    snap = Snapshot(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=0,
        blocks=(
            SnapshotBlock(
                sequence_hash=chain[0].sequence_hash,
                parent_sequence_hash=None,
                block_hash=chain[0].block_hash,
                tiers=("device",),
            ),
        ),
    )

    pulls = 0

    async def pull_fn(*args):
        nonlocal pulls
        pulls += 1
        return snap

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        rec.register_target(publisher_id="w1", endpoint="ignored", model="m", compat_key="ck")
        await rec.start()
        # Allow the initial periodic sweep to land (registered target).
        await asyncio.sleep(0.05)
        baseline = pulls

        rec.trigger_gap_recovery(publisher_id="w1", endpoint="ignored", model="m", compat_key="ck")
        # Should pick this up within a short period.
        await asyncio.sleep(0.05)
        assert pulls > baseline
    finally:
        await rec.stop()
        await writer.stop()


async def test_reconciler_full_loop_with_producer() -> None:
    """Realistic flow: probe writes events to producer; reconciler later
    pulls a snapshot from the producer and applies to an initially-empty
    index. End state should match what the events would have applied."""
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    chain = hash_token_blocks(list(range(8)), block_size=4)
    for batch_id, block in enumerate(chain):
        producer.on_event(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=block.sequence_hash,
                block_hash=block.block_hash,
                parent_sequence_hash=block.parent_sequence_hash,
                tier="device",
            ),
            batch_id=batch_id,
        )

    async def pull_fn(pub_id, ep, model, ck):
        return producer.snapshot(model=model, compat_key=ck)

    index, writer, rec = await _make_reconciler(pull_fn)
    try:
        await rec.reconcile_now(publisher_id="w1", endpoint="ignored", model="m", compat_key="ck")
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks(device=2)
    finally:
        await rec.stop()
        await writer.stop()


# ----------------------------------------------------------------------
# KVIndex.drop_tree
# ----------------------------------------------------------------------


def test_index_drop_tree() -> None:
    index = KVIndex()
    chain = hash_token_blocks(list(range(4)), block_size=4)
    index.apply_stored(
        publisher_id="w1",
        model="m",
        compat_key="ckA",
        block=chain[0],
        tier=Tier.DEVICE,
    )
    index.apply_stored(
        publisher_id="w1",
        model="m",
        compat_key="ckB",
        block=chain[0],
        tier=Tier.DEVICE,
    )
    index.drop_tree(model="m", compat_key="ckA", publisher_id="w1")
    # ckA gone, ckB intact.
    matches_a = index.find_matches(model="m", compat_key="ckA", chain=[chain[0]], candidates=["w1"])
    matches_b = index.find_matches(model="m", compat_key="ckB", chain=[chain[0]], candidates=["w1"])
    assert matches_a["w1"] == OverlapBlocks()
    assert matches_b["w1"].device == 1
