###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""HTTP-surface tests for the KV management endpoints + pull function.

Uses httpx.ASGITransport to talk to FastAPI routers in-process — no
real network. The snapshot pull function is then tested by wiring it
to call against an ASGI-mounted worker app, so the whole pipe
(reconciler → HTTP → snapshot endpoint → producer) is exercised
end-to-end without external services.
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from infera.kv.api import (
    make_http_snapshot_puller,
    make_snapshot_router,
    make_stats_router,
)
from infera.kv.hashing import hash_token_blocks
from infera.kv.index import KVIndex
from infera.kv.snapshot import SnapshotProducer, SnapshotReconciler
from infera.kv.subscriber import KvEventSubscriberPool
from infera.kv.types import OverlapBlocks
from infera.kv.wire import make_stored_event
from infera.kv.writer import KvIndexWriter

# ----------------------------------------------------------------------
# Snapshot router (worker side)
# ----------------------------------------------------------------------


async def test_snapshot_endpoint_returns_empty_when_no_events() -> None:
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    app = FastAPI()
    app.include_router(make_snapshot_router(producer))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/kv-snapshot", params={"model": "m", "compat_key": "ck"})
    assert resp.status_code == 200
    obj = resp.json()
    assert obj["publisher_id"] == "w1"
    assert obj["publisher_type"] == "worker"
    assert obj["model_name"] == "m"
    assert obj["compat_key"] == "ck"
    assert obj["index_block_size"] == 4
    assert obj["batch_id"] == -1  # nothing emitted yet
    assert obj["blocks"] == []


async def test_snapshot_endpoint_returns_recorded_blocks() -> None:
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

    app = FastAPI()
    app.include_router(make_snapshot_router(producer))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/kv-snapshot", params={"model": "m", "compat_key": "ck"})
    obj = resp.json()
    assert len(obj["blocks"]) == 2
    assert obj["batch_id"] == 1  # highest batch_id seen


async def test_snapshot_endpoint_requires_model_param() -> None:
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    app = FastAPI()
    app.include_router(make_snapshot_router(producer))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/kv-snapshot", params={"compat_key": "ck"})
    assert resp.status_code == 422  # FastAPI validation error


# ----------------------------------------------------------------------
# Stats router (server side)
# ----------------------------------------------------------------------


async def test_stats_endpoint_index_only() -> None:
    """All optional sections are omitted when components aren't provided."""
    index = KVIndex()
    app = FastAPI()
    app.include_router(make_stats_router(index=index))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/kv-stats")
    obj = resp.json()
    assert "index" in obj
    assert obj["index"]["nodes"] == 0
    assert "writer" not in obj
    assert "subscribers" not in obj
    assert "reconciler" not in obj


async def test_stats_endpoint_with_all_components() -> None:
    index = KVIndex()
    queue: asyncio.Queue = asyncio.Queue()
    writer = KvIndexWriter(index=index, queue=queue)
    await writer.start()
    subs = KvEventSubscriberPool(output_queue=queue)
    rec = SnapshotReconciler(
        index=index,
        writer=writer,
        pull_fn=lambda *args, **kwargs: None,  # never called
        interval_s=10_000,
    )

    app = FastAPI()
    app.include_router(
        make_stats_router(index=index, writer=writer, subscribers=subs, reconciler=rec)
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/kv-stats")
        obj = resp.json()
        assert "writer" in obj
        assert obj["writer"]["batches_applied"] == 0
        assert "subscribers" in obj
        assert obj["subscribers"]["endpoints"] == []
        assert "reconciler" in obj
        assert obj["reconciler"]["snapshots_pulled"] == 0
    finally:
        await writer.stop()


async def test_stats_endpoint_reflects_writer_activity() -> None:
    index = KVIndex()
    queue: asyncio.Queue = asyncio.Queue()
    writer = KvIndexWriter(index=index, queue=queue)
    await writer.start()
    # Bump writer counters by feeding it a synthetic batch.
    from infera.kv.wire import EventBatch

    batch = EventBatch(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=4,
        batch_id=0,
        events=(
            make_stored_event(
                sequence_hash=1, block_hash=2, parent_sequence_hash=None, tier="device"
            ),
        ),
    )
    await queue.put(batch)
    await asyncio.sleep(0.2)

    app = FastAPI()
    app.include_router(make_stats_router(index=index, writer=writer))
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/kv-stats")
        obj = resp.json()
        assert obj["writer"]["events_applied_stored"] == 1
        assert obj["index"]["nodes"] == 1
    finally:
        await writer.stop()


# ----------------------------------------------------------------------
# HTTP snapshot pull function
# ----------------------------------------------------------------------


async def test_http_snapshot_puller_round_trip() -> None:
    """Mount the snapshot router on an ASGI app and verify the puller
    can fetch and decode."""
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    chain = hash_token_blocks(list(range(8)), block_size=4)
    for block in chain:
        producer.on_event(
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

    app = FastAPI()
    app.include_router(make_snapshot_router(producer))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pull = make_http_snapshot_puller(client=client)
        snapshot = await pull("w1", "http://test", "m", "ck")
    assert snapshot is not None
    assert len(snapshot.blocks) == 2
    assert snapshot.blocks[0].sequence_hash == chain[0].sequence_hash


async def test_http_snapshot_puller_returns_none_on_404() -> None:
    """Endpoint not mounted at the expected path → puller returns None."""
    app = FastAPI()  # no routes
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pull = make_http_snapshot_puller(client=client)
        snapshot = await pull("w1", "http://test", "m", "ck")
    assert snapshot is None


async def test_http_snapshot_puller_returns_none_on_bad_json() -> None:
    """Endpoint returns garbage → puller returns None gracefully."""
    app = FastAPI()

    @app.get("/v1/kv-snapshot")
    async def bad():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("not json")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pull = make_http_snapshot_puller(client=client)
        snapshot = await pull("w1", "http://test", "m", "ck")
    assert snapshot is None


async def test_http_snapshot_puller_full_pipeline_to_reconciler() -> None:
    """End-to-end: worker producer → /v1/kv-snapshot → http puller →
    SnapshotReconciler → KVIndex."""
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    chain = hash_token_blocks(list(range(8)), block_size=4)
    for block in chain:
        producer.on_event(
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

    worker_app = FastAPI()
    worker_app.include_router(make_snapshot_router(producer))
    transport = httpx.ASGITransport(app=worker_app)

    index = KVIndex()
    queue: asyncio.Queue = asyncio.Queue()
    writer = KvIndexWriter(index=index, queue=queue)
    await writer.start()
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pull = make_http_snapshot_puller(client=client)
            rec = SnapshotReconciler(index=index, writer=writer, pull_fn=pull, interval_s=10_000)
            ok = await rec.reconcile_now(
                publisher_id="w1", endpoint="http://test", model="m", compat_key="ck"
            )
        assert ok is True
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks(device=2)
    finally:
        await writer.stop()
