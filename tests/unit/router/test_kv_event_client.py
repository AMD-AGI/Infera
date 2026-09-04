###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/router/kv_event/client.py.

Two layers:
  - State-machine layer: exercise `_handle_event` directly against
    handcrafted `BlockStored / BlockRemoved / AllBlocksCleared` payloads.
    These are pure-Python unit tests, no sockets.
  - Transport layer: spin up a real ZMQ PUB socket, msgspec-encode a
    `SglangKVEventBatch` exactly the way SGLang does, let `KvEventClient.on_worker_added`
    subscribe, and assert the resulting cache_view. Catches wire-format
    drift between Infera and upstream.

The `cache_view` membership predicate is what the policy queries, so
that's what the tests check — not internal map/view dicts.
"""

from __future__ import annotations

import asyncio
import socket

import msgspec
import pytest
import zmq
import zmq.asyncio

from infera.common.worker_pool import (
    DisaggMode,
    EngineType,
    WorkerInfo,
    WorkerStatus,
)
from infera.router.kv_event.client import (
    KvEventClient,
    WorkerSubscription,
)
from infera.router.kv_event.events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    SglangBlockStored,
    SglangKVEventBatch,
    batch_type_for_engine,
)
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_info(
    worker_id: str = "w1",
    *,
    kv_endpoint: str | None = "tcp://127.0.0.1:5555",
    kv_block_size: int = 4,
) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url="http://127.0.0.1:8000",
        model_name="test/m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=kv_endpoint,
        kv_block_size=kv_block_size,
    )


def _make_stored(
    *,
    block_hashes: list[int],
    parent_block_hash: int | None,
    token_ids: list[int],
    block_size: int = 4,
    lora_id: int | None = None,
    medium: str | None = "device",
) -> BlockStored:
    return BlockStored(
        block_hashes=block_hashes,
        parent_block_hash=parent_block_hash,
        token_ids=token_ids,
        block_size=block_size,
        lora_id=lora_id,
        medium=medium,
    )


def _make_sglang_stored(
    *,
    block_hashes: list[int],
    parent_block_hash: int | None,
    token_ids: list[int],
    block_size: int = 4,
    lora_id: int | None = None,
    medium: str | None = "device",
) -> SglangBlockStored:
    """SGLang wire format (tagged ARRAY event) — what a SGLANG-engine worker
    actually emits, so the transport tests below (whose worker is SGLANG) must
    encode with this, not the vLLM tagged-MAP `BlockStored`."""
    return SglangBlockStored(
        block_hashes=block_hashes,
        parent_block_hash=parent_block_hash,
        token_ids=token_ids,
        block_size=block_size,
        lora_id=lora_id,
        medium=medium,
    )


# ----------------------------------------------------------------------
# State-machine layer: _handle_event
# ----------------------------------------------------------------------


def test_block_stored_chains_from_seed_when_no_parent():
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    ev = _make_stored(
        block_hashes=[111],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
    )
    client._handle_event(sub, ev)

    expected = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
    assert sub.view_for(None) == {expected}
    assert sub.map_for(None) == {111: expected}


def test_block_stored_chains_multiple_blocks_in_one_event():
    """SGLang sometimes coalesces several blocks into one event;
    block_hashes is len-aligned with token_ids // block_size."""
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    ev = _make_stored(
        block_hashes=[111, 222],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    client._handle_event(sub, ev)

    h0 = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
    h1 = hash_chunk(h0, [5, 6, 7, 8])
    assert sub.view_for(None) == {h0, h1}
    assert sub.map_for(None) == {111: h0, 222: h1}


def test_block_stored_chains_from_parent_when_present():
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    # Block 0: worker says "no parent"
    client._handle_event(
        sub,
        _make_stored(
            block_hashes=[111],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4],
        ),
    )
    h0 = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])

    # Block 1: worker says "parent is block 111"
    client._handle_event(
        sub,
        _make_stored(
            block_hashes=[222],
            parent_block_hash=111,
            token_ids=[5, 6, 7, 8],
        ),
    )
    h1 = hash_chunk(h0, [5, 6, 7, 8])

    assert sub.view_for(None) == {h0, h1}
    assert sub.map_for(None)[222] == h1


def test_block_stored_drops_when_parent_unknown():
    """Missing-parent: silently drop so a future cold-start window
    doesn't crash. Worst case is one cache miss."""
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    client._handle_event(
        sub,
        _make_stored(
            block_hashes=[222],
            parent_block_hash=999,  # never seen
            token_ids=[5, 6, 7, 8],
        ),
    )

    assert sub.view_for(None) == set()
    assert sub.map_for(None) == {}


def test_block_removed_evicts_matching_hashes():
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    client._handle_event(
        sub,
        _make_stored(
            block_hashes=[111, 222],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        ),
    )
    h0 = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
    h1 = hash_chunk(h0, [5, 6, 7, 8])
    assert sub.view_for(None) == {h0, h1}

    client._handle_event(sub, BlockRemoved(block_hashes=[222]))
    assert sub.view_for(None) == {h0}
    assert sub.map_for(None) == {111: h0}


def test_block_removed_ignores_unknown_hash():
    """Worker may emit a Removed for a block whose Stored we never
    saw (replay during reconnect). Don't crash."""
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    client._handle_event(sub, BlockRemoved(block_hashes=[777, 888]))
    assert sub.view_for(None) == set()
    assert sub.map_for(None) == {}


def test_all_blocks_cleared_resets_view_and_map():
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)
    client._handle_event(
        sub,
        _make_stored(
            block_hashes=[111, 222],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        ),
    )
    assert sub.view_for(None)  # populated

    client._handle_event(sub, AllBlocksCleared())
    assert sub.view_for(None) == set()
    assert sub.map_for(None) == {}


def test_block_stored_zero_tokens_is_safe():
    """Edge: token_ids // block_size == 0 → loop runs 0 times."""
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    client._handle_event(
        sub,
        _make_stored(block_hashes=[], parent_block_hash=None, token_ids=[]),
    )
    assert sub.view_for(None) == set()
    assert sub.map_for(None) == {}


# ----------------------------------------------------------------------
# Public API: cache_view, on_worker_added, on_worker_removed
# ----------------------------------------------------------------------


def test_cache_view_empty_for_unknown_worker():
    client = KvEventClient()
    assert client.cache_view("nobody") == set()


def test_on_worker_added_ignores_workers_without_kv_endpoint():
    client = KvEventClient()
    w = _worker_info(kv_endpoint=None)
    client.on_worker_added(w)
    assert client._subs == {}


def test_on_worker_added_is_idempotent():
    """Re-adding the same worker_id shouldn't start a second subscriber task."""

    async def run():
        client = KvEventClient()
        port = _free_port()
        w = _worker_info(kv_endpoint=f"tcp://127.0.0.1:{port}")
        client.on_worker_added(w)
        client.on_worker_added(w)
        try:
            assert len(client._subs) == 1
        finally:
            await client.aclose()

    asyncio.run(run())


def test_on_worker_removed_cancels_subscriber_task():
    async def run():
        client = KvEventClient()
        port = _free_port()
        w = _worker_info(kv_endpoint=f"tcp://127.0.0.1:{port}")
        client.on_worker_added(w)
        tasks = client._subs[w.worker_id].tasks
        assert len(tasks) == 1
        task = tasks[0]

        client.on_worker_removed(w.worker_id)
        assert w.worker_id not in client._subs
        # Give the cancel one tick to propagate.
        await asyncio.sleep(0.05)
        assert task.cancelled() or task.done()
        await client.aclose()

    asyncio.run(run())


def test_on_worker_removed_unknown_id_is_noop():
    client = KvEventClient()
    client.on_worker_removed("ghost")  # must not raise


# ----------------------------------------------------------------------
# Wire-format / transport layer
# ----------------------------------------------------------------------


def test_atom_decoder_matches_infera_atom_hook_wire_format():
    """ATOM's Infera hook publishes the same tagged-map schema as vLLM.

    Selecting SGLang's tagged-array decoder for an ATOM worker rejects every
    event and leaves KV-aware routing with an empty cache view.
    """
    payload = msgspec.msgpack.encode(
        KVEventBatch(
            ts=1.0,
            events=[
                _make_stored(
                    block_hashes=[111],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4],
                )
            ],
        )
    )

    decoded = msgspec.msgpack.decode(payload, type=batch_type_for_engine(EngineType.ATOM))

    assert isinstance(decoded, KVEventBatch)
    assert isinstance(decoded.events[0], BlockStored)
    assert decoded.events[0].block_hashes == [111]


@pytest.mark.asyncio
async def test_real_zmq_publisher_drives_cache_view():
    """End-to-end: a real ZMQ PUB on `tcp://127.0.0.1:<port>` emits a
    msgspec-encoded SglangKVEventBatch matching SGLang's wire format. The
    KvEventClient must subscribe, decode, and update cache_view.
    """
    port = _free_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    topic = b"kv-events"

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)

    # PUB-SUB late-joiner problem: a brand-new SUB that connects after
    # PUB has already sent is fine, but messages sent BEFORE the subscriber
    # is filtering the topic are lost. We give the subscriber a moment to
    # attach and add the topic filter.
    client = KvEventClient()
    w = _worker_info(kv_endpoint=endpoint, kv_block_size=4)
    client.on_worker_added(w)

    try:
        encoder = msgspec.msgpack.Encoder()
        batch = SglangKVEventBatch(
            ts=1.0,
            events=[
                _make_sglang_stored(
                    block_hashes=[111, 222],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                ),
            ],
        )
        payload = encoder.encode(batch)

        # Re-send a few times to defeat the late-joiner race deterministically.
        deadline = asyncio.get_running_loop().time() + 3.0
        h0 = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
        h1 = hash_chunk(h0, [5, 6, 7, 8])
        expected = {h0, h1}
        while asyncio.get_running_loop().time() < deadline:
            pub.send_multipart([topic, payload])
            await asyncio.sleep(0.05)
            if client.cache_view(w.worker_id) == expected:
                break
        else:
            pytest.fail(f"cache_view never reached expected; got {client.cache_view(w.worker_id)}")
    finally:
        pub.close(linger=0)
        ctx.term()
        await client.aclose()


@pytest.mark.asyncio
async def test_subscriber_decode_error_does_not_kill_loop():
    """A malformed msgpack frame should be logged + skipped; the loop
    must keep running and process the next valid batch."""
    port = _free_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    topic = b"kv-events"

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)

    client = KvEventClient()
    w = _worker_info(kv_endpoint=endpoint, kv_block_size=4)
    client.on_worker_added(w)

    try:
        encoder = msgspec.msgpack.Encoder()
        good = encoder.encode(
            SglangKVEventBatch(
                ts=1.0,
                events=[
                    _make_sglang_stored(
                        block_hashes=[111],
                        parent_block_hash=None,
                        token_ids=[1, 2, 3, 4],
                    ),
                ],
            )
        )

        h0 = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
        deadline = asyncio.get_running_loop().time() + 3.0
        while asyncio.get_running_loop().time() < deadline:
            pub.send_multipart([topic, b"\xff\xff\xff garbage"])
            pub.send_multipart([topic, good])
            await asyncio.sleep(0.05)
            if h0 in client.cache_view(w.worker_id):
                break
        else:
            pytest.fail("subscriber didn't recover from a bad frame")
    finally:
        pub.close(linger=0)
        ctx.term()
        await client.aclose()


# ---- re-registration in place ---------------------------------------------
#
# `sync`/discovery only report a departure when the worker actually leaves. A
# stop that skips the SIGTERM handler -- SIGKILL, OOM, node loss -- never runs
# `deregister`, so the k8s annotation is never cleared, the Pod holds
# `phase: Running` across the container restart, and the replacement process
# republishes the same worker_id with the fresh `free_tcp_port()` its
# `kv_events_endpoint` just took. Confirmed live on the 1P1D GLM-5.3 fleet:
# rewriting that one annotation field logged `k8s: worker ... updated`, never
# `removed`. Mirror of `a_killed_worker_re_registers_in_place` in
# rust/router/src/kv_event.rs.


def _w(endpoint: str | None = "tcp://10.0.0.1:5557", block_size: int | None = 64) -> WorkerInfo:
    return WorkerInfo(
        worker_id="10.0.0.1:30000",
        url="http://10.0.0.1:30000",
        model_name="m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.PREFILL,
        kv_events_endpoint=endpoint,
        kv_block_size=block_size,
    )


@pytest.mark.asyncio
async def test_a_replacement_process_on_the_same_worker_id_is_resubscribed():
    c = KvEventClient()
    try:
        c.on_worker_added(_w())
        c._subs["10.0.0.1:30000"].view_for(0).update({11, 22, 33})
        assert len(c.cache_view("10.0.0.1:30000")) == 3, "the warm engine's view"

        # Same Pod, same worker_id, new port -- no departure in between.
        c.on_worker_added(_w(endpoint="tcp://10.0.0.1:41907"))

        assert c._subs["10.0.0.1:30000"].endpoint == "tcp://10.0.0.1:41907", (
            "the dead socket must not outlive the process that opened it"
        )
        assert c.cache_view("10.0.0.1:30000") == set(), (
            "the engine restarted with an empty prefix cache; a view carried over "
            "from before it makes the router score hits that cannot exist"
        )
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_an_unchanged_republish_keeps_the_view():
    """Discovery republishes on every heartbeat annotation refresh. Re-deciding
    then would throw the KV view away several times a minute."""
    c = KvEventClient()
    try:
        c.on_worker_added(_w())
        c._subs["10.0.0.1:30000"].view_for(0).update({1, 2, 3, 4, 5})
        for _ in range(5):
            c.on_worker_added(_w())
        assert len(c.cache_view("10.0.0.1:30000")) == 5
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_a_corrected_block_size_is_picked_up_in_place():
    """A worker declined for having no block size, then fixed without leaving."""
    c = KvEventClient()
    try:
        c.on_worker_added(_w(block_size=None))
        assert "10.0.0.1:30000" not in c._subs
        c.on_worker_added(_w(block_size=64))
        assert c._subs["10.0.0.1:30000"].block_size == 64
    finally:
        await c.aclose()
