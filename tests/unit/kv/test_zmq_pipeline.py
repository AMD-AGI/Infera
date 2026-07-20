###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end pipeline test: publisher → ZMQ → subscriber → writer → KVIndex.

Uses ZMQ `inproc://` transport so no network. asyncio-mode tests
(configured in pyproject.toml: `asyncio_mode = "auto"`).
"""

from __future__ import annotations

import asyncio

from infera.kv.hashing import hash_token_blocks
from infera.kv.index import KVIndex
from infera.kv.publisher import KvEventPublisher
from infera.kv.subscriber import KvEventSubscriber, KvEventSubscriberPool
from infera.kv.types import OverlapBlocks
from infera.kv.wire import (
    EventBatch,
    NormalizedEvent,
    make_cleared_event,
    make_removed_event,
    make_stored_event,
)
from infera.kv.writer import KvIndexWriter

# Distinct endpoint per test to avoid ZMQ context state leaking across tests.
_endpoint_counter = 0


def _next_endpoint() -> str:
    global _endpoint_counter
    _endpoint_counter += 1
    return f"inproc://infera-kv-test-{_endpoint_counter}"


# ----------------------------------------------------------------------
# Publisher → Subscriber (transport-only)
# ----------------------------------------------------------------------


async def test_publisher_to_subscriber_basic() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=10,
        batch_max_ms=50,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await pub.start()
    await sub.start()
    try:
        # Give SUB a moment to be ready (inproc still has slow-joiner semantics).
        await asyncio.sleep(0.05)

        e = make_stored_event(
            sequence_hash=11, block_hash=12, parent_sequence_hash=None, tier="device"
        )
        pub.emit(model="m", compat_key="ck", event=e)
        # Wait for the timer-driven flush.
        batch = await asyncio.wait_for(queue.get(), timeout=2.0)
    finally:
        await pub.stop()
        await sub.stop()

    assert batch.publisher_id == "w1"
    assert batch.model_name == "m"
    assert batch.compat_key == "ck"
    assert batch.batch_id == 0
    assert len(batch.events) == 1
    assert batch.events[0].sequence_hash == 11


async def test_publisher_size_triggers_immediate_flush() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=3,
        batch_max_ms=10_000,  # huge timer; only size-trigger should fire
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await pub.start()
    await sub.start()
    try:
        await asyncio.sleep(0.05)
        for i in range(3):
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=i, block_hash=i + 100, parent_sequence_hash=None, tier="device"
                ),
            )
        batch = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert len(batch.events) == 3
    finally:
        await pub.stop()
        await sub.stop()


async def test_publisher_batch_id_monotonic_per_stream() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=2,
        batch_max_ms=10_000,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await pub.start()
    await sub.start()
    try:
        await asyncio.sleep(0.05)
        # Two batches of 2 events each.
        for i in range(4):
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=i, block_hash=i + 100, parent_sequence_hash=None, tier="device"
                ),
            )
        b1 = await asyncio.wait_for(queue.get(), timeout=2.0)
        b2 = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert b1.batch_id == 0
        assert b2.batch_id == 1
    finally:
        await pub.stop()
        await sub.stop()


async def test_publisher_separate_streams_per_model_compat() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=1,
        batch_max_ms=10_000,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await pub.start()
    await sub.start()
    try:
        await asyncio.sleep(0.05)
        pub.emit(
            model="mA",
            compat_key="ckA",
            event=make_stored_event(
                sequence_hash=1, block_hash=10, parent_sequence_hash=None, tier="device"
            ),
        )
        pub.emit(
            model="mB",
            compat_key="ckB",
            event=make_stored_event(
                sequence_hash=2, block_hash=20, parent_sequence_hash=None, tier="device"
            ),
        )
        b1 = await asyncio.wait_for(queue.get(), timeout=2.0)
        b2 = await asyncio.wait_for(queue.get(), timeout=2.0)
        seen = {(b.model_name, b.compat_key): b for b in (b1, b2)}
        assert ("mA", "ckA") in seen
        assert ("mB", "ckB") in seen
        # Each stream starts at batch_id 0.
        assert seen[("mA", "ckA")].batch_id == 0
        assert seen[("mB", "ckB")].batch_id == 0
    finally:
        await pub.stop()
        await sub.stop()


# ----------------------------------------------------------------------
# Subscriber: drop-on-overload metric
# ----------------------------------------------------------------------


async def test_subscriber_drops_when_queue_full() -> None:
    endpoint = _next_endpoint()
    # Queue size 1; we send 5 batches.
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=1,
        batch_max_ms=10_000,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await pub.start()
    await sub.start()
    try:
        await asyncio.sleep(0.05)
        for i in range(5):
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=i, block_hash=i + 1, parent_sequence_hash=None, tier="device"
                ),
            )
        # Let publisher batch + subscriber drain attempts complete.
        await asyncio.sleep(0.5)
        # One batch in queue, others dropped.
        assert queue.qsize() == 1
        assert sub.metrics.batches_dropped_overload >= 1
    finally:
        await pub.stop()
        await sub.stop()


async def test_subscriber_drops_malformed_payload() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    await sub.start()

    # Open a manual PUB to send garbage.
    import zmq
    import zmq.asyncio

    ctx = zmq.asyncio.Context.instance()
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.bind(endpoint)
    try:
        await asyncio.sleep(0.05)
        await pub_sock.send_multipart([b"m", b"\x00\x01\x02not-msgpack"])
        await asyncio.sleep(0.2)
        assert sub.metrics.batches_dropped_malformed >= 1
        assert queue.qsize() == 0
    finally:
        pub_sock.close(linger=0)
        await sub.stop()


# ----------------------------------------------------------------------
# End-to-end: publisher → subscriber → writer → KVIndex
# ----------------------------------------------------------------------


async def test_end_to_end_stored_lands_in_index() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    index = KVIndex()
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=4,  # match the test hash chain
        batch_max_events=10,
        batch_max_ms=50,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    writer = KvIndexWriter(index=index, queue=queue)
    await pub.start()
    await sub.start()
    await writer.start()
    try:
        await asyncio.sleep(0.05)
        chain = hash_token_blocks(list(range(8)), block_size=4)
        for block in chain:
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=block.sequence_hash,
                    block_hash=block.block_hash,
                    parent_sequence_hash=block.parent_sequence_hash,
                    tier="device",
                ),
            )
        # Wait for the pipeline to settle.
        await asyncio.sleep(0.3)
        # Index now has the chain.
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks(device=2)
    finally:
        await writer.stop()
        await sub.stop()
        await pub.stop()


async def test_end_to_end_removed_clears_index() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    index = KVIndex()
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=4,
        batch_max_events=10,
        batch_max_ms=50,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    writer = KvIndexWriter(index=index, queue=queue)
    await pub.start()
    await sub.start()
    await writer.start()
    try:
        await asyncio.sleep(0.05)
        chain = hash_token_blocks(list(range(8)), block_size=4)
        # First: store both blocks
        for block in chain:
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=block.sequence_hash,
                    block_hash=block.block_hash,
                    parent_sequence_hash=block.parent_sequence_hash,
                    tier="device",
                ),
            )
        await asyncio.sleep(0.2)
        # Then: remove the second block.
        pub.emit(
            model="m",
            compat_key="ck",
            event=make_removed_event(sequence_hash=chain[1].sequence_hash, tier="device"),
        )
        await asyncio.sleep(0.2)
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        # Only the first block remains.
        assert matches["w1"].device == 1
    finally:
        await writer.stop()
        await sub.stop()
        await pub.stop()


async def test_end_to_end_cleared_drops_compat_key() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    index = KVIndex()
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=4,
        batch_max_events=10,
        batch_max_ms=50,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    writer = KvIndexWriter(index=index, queue=queue)
    await pub.start()
    await sub.start()
    await writer.start()
    try:
        await asyncio.sleep(0.05)
        chain = hash_token_blocks(list(range(8)), block_size=4)
        for block in chain:
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=block.sequence_hash,
                    block_hash=block.block_hash,
                    parent_sequence_hash=block.parent_sequence_hash,
                    tier="device",
                ),
            )
        await asyncio.sleep(0.2)
        pub.emit(
            model="m",
            compat_key="ck",
            event=make_cleared_event(scope="compat_key:ck"),
        )
        await asyncio.sleep(0.2)
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks()
    finally:
        await writer.stop()
        await sub.stop()
        await pub.stop()


async def test_subscriber_pool_add_remove() -> None:
    endpoint_a = _next_endpoint()
    endpoint_b = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    pub_a = KvEventPublisher(
        bind_endpoint=endpoint_a,
        publisher_id="wA",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=1,
        batch_max_ms=10_000,
    )
    pub_b = KvEventPublisher(
        bind_endpoint=endpoint_b,
        publisher_id="wB",
        publisher_type="worker",
        index_block_size=64,
        batch_max_events=1,
        batch_max_ms=10_000,
    )
    pool = KvEventSubscriberPool(output_queue=queue)
    await pub_a.start()
    await pub_b.start()
    try:
        await pool.add(endpoint_a)
        await pool.add(endpoint_b)
        await asyncio.sleep(0.1)

        pub_a.emit(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=1, block_hash=11, parent_sequence_hash=None, tier="device"
            ),
        )
        pub_b.emit(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=2, block_hash=22, parent_sequence_hash=None, tier="device"
            ),
        )
        b1 = await asyncio.wait_for(queue.get(), timeout=2.0)
        b2 = await asyncio.wait_for(queue.get(), timeout=2.0)
        publisher_ids = {b1.publisher_id, b2.publisher_id}
        assert publisher_ids == {"wA", "wB"}

        # Remove subscriber B; further emits from B should not arrive.
        await pool.remove(endpoint_b)
        pub_a.emit(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=3, block_hash=33, parent_sequence_hash=None, tier="device"
            ),
        )
        pub_b.emit(
            model="m",
            compat_key="ck",
            event=make_stored_event(
                sequence_hash=4, block_hash=44, parent_sequence_hash=None, tier="device"
            ),
        )
        # Drain anything that arrives in 200 ms.
        received = []
        end = asyncio.get_event_loop().time() + 0.3
        while asyncio.get_event_loop().time() < end:
            try:
                b = await asyncio.wait_for(queue.get(), timeout=0.1)
                received.append(b)
            except asyncio.TimeoutError:
                break
        # Only batches from wA should appear.
        assert all(b.publisher_id == "wA" for b in received)
        assert len(received) >= 1
    finally:
        await pool.stop_all()
        await pub_a.stop()
        await pub_b.stop()


# ----------------------------------------------------------------------
# Writer-side metrics
# ----------------------------------------------------------------------


async def test_writer_metrics_counts() -> None:
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    index = KVIndex()
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=4,
        batch_max_events=10,
        batch_max_ms=50,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    writer = KvIndexWriter(index=index, queue=queue)
    await pub.start()
    await sub.start()
    await writer.start()
    try:
        await asyncio.sleep(0.05)
        chain = hash_token_blocks(list(range(8)), block_size=4)
        for block in chain:
            pub.emit(
                model="m",
                compat_key="ck",
                event=make_stored_event(
                    sequence_hash=block.sequence_hash,
                    block_hash=block.block_hash,
                    parent_sequence_hash=block.parent_sequence_hash,
                    tier="device",
                ),
            )
        pub.emit(
            model="m",
            compat_key="ck",
            event=make_removed_event(sequence_hash=chain[0].sequence_hash, tier="device"),
        )
        await asyncio.sleep(0.3)
        assert writer.metrics.events_applied_stored == 2
        assert writer.metrics.events_applied_removed == 1
        assert writer.metrics.batches_applied >= 1
    finally:
        await writer.stop()
        await sub.stop()
        await pub.stop()


async def test_writer_drops_malformed_event() -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    index = KVIndex()
    writer = KvIndexWriter(index=index, queue=queue)
    await writer.start()
    try:
        # Directly feed a batch with a malformed stored event (missing tier).
        bad_batch = EventBatch(
            publisher_id="w1",
            publisher_type="worker",
            model_name="m",
            compat_key="ck",
            index_block_size=4,
            batch_id=0,
            events=(
                NormalizedEvent(
                    type=__import__("infera.kv.wire", fromlist=["EventType"]).EventType.STORED,
                    sequence_hash=1,
                    block_hash=2,
                    parent_sequence_hash=None,
                    # tier intentionally missing
                ),
            ),
        )
        await queue.put(bad_batch)
        await asyncio.sleep(0.2)
        assert writer.metrics.events_dropped_malformed == 1
        assert writer.metrics.events_applied_stored == 0
    finally:
        await writer.stop()
