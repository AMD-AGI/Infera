###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end test for the kv-aware routing pipeline on main.

Wires a fake ZMQ publisher (matching SGLang's msgspec wire format),
to a real `KvEventClient`, attached to a real `KvEventAwarePolicy`.
Asserts that after BlockStored events drive the cache view, the policy
picks the worker whose cache aligns with the request.

This is the contract test that locks in:
  - wire format byte-for-byte compatibility with SGLang/vLLM
    (the events module's msgspec config: array_like, tag=True, etc.)
  - chained-hash translation worker_hash → router_hash via map
  - request hashing aligns with event hashing (both use
    `hash_request(token_ids, block_size)` with `ROUTER_SEED=0`)

The "request tokens line up with worker tokens" guarantee is what
makes a cache HIT possible. Test #2 explicitly breaks that alignment
to verify the miss path.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import msgspec
import pytest
import zmq

from infera.common.worker_pool import (
    DisaggMode,
    EngineType,
    WorkerInfo,
    WorkerStatus,
)
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.events import SglangBlockStored, SglangKVEventBatch
from infera.router.policy.kv_event_aware import KvEventAwarePolicy
from infera.router.policy.target import RouteTarget

_TOPIC = b"kv-events"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(worker_id: str, *, endpoint: str | None = None, kv_block_size: int = 4) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url=f"http://{worker_id}",
        model_name="test/m",
        engine=EngineType.SGLANG,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=endpoint,
        kv_block_size=kv_block_size,
    )


class _IdentityHasher:
    """Hashes a request by treating `body["token_ids"]` as a pre-tokenized
    list — exactly what the worker fed into its KV events. This lets the
    test control the alignment between request and worker tokens
    deterministically, decoupled from any real tokenizer."""

    def __init__(self, block_size: int = 4) -> None:
        self._block_size = block_size

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        from infera.router.kv_event.hasher import hash_request

        token_ids = body.get("token_ids", [])
        return hash_request(token_ids, block_size)


async def _publish_until_visible(
    pub: Any,
    payload: bytes,
    predicate,
    *,
    deadline_s: float = 3.0,
    interval_s: float = 0.05,
) -> bool:
    """PUB/SUB has a late-joiner race; re-send periodically and poll
    `predicate()` until it's true or the deadline expires."""
    deadline = asyncio.get_running_loop().time() + deadline_s
    while asyncio.get_running_loop().time() < deadline:
        pub.send_multipart([_TOPIC, payload])
        await asyncio.sleep(interval_s)
        if predicate():
            return True
    return False


@pytest.mark.asyncio
async def test_full_pipeline_routes_to_worker_with_cached_prefix():
    """Two workers, one fake publisher per worker. Worker A emits
    BlockStored events that cover the request prefix; Worker B emits
    nothing. The policy must pick Worker A."""

    port_a = _free_port()
    port_b = _free_port()
    ep_a = f"tcp://127.0.0.1:{port_a}"
    ep_b = f"tcp://127.0.0.1:{port_b}"

    ctx = zmq.Context()
    pub_a = ctx.socket(zmq.PUB)
    pub_a.bind(ep_a)
    pub_b = ctx.socket(zmq.PUB)
    pub_b.bind(ep_b)

    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())

    w_a = _worker("wA", endpoint=ep_a, kv_block_size=4)
    w_b = _worker("wB", endpoint=ep_b, kv_block_size=4)
    policy.on_worker_added(w_a)
    policy.on_worker_added(w_b)

    try:
        # Worker A says "I cached two blocks of [1..4, 5..8]"
        encoder = msgspec.msgpack.Encoder()
        batch = SglangKVEventBatch(
            ts=1.0,
            events=[
                SglangBlockStored(
                    block_hashes=[1001, 1002],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    block_size=4,
                    lora_id=None,
                    medium="device",
                ),
            ],
        )
        payload = encoder.encode(batch)

        ok = await _publish_until_visible(
            pub_a,
            payload,
            predicate=lambda: len(client.cache_view("wA")) == 2,
            deadline_s=3.0,
        )
        assert ok, f"wA view never populated; got {client.cache_view('wA')}"

        # Request asks for the exact 8-token prefix wA cached.
        picked, blocks = policy.pick(
            [w_a, w_b],
            {"model": "test/m", "token_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
        )
        assert picked.worker.worker_id == "wA"
        assert len(blocks) == 2
    finally:
        pub_a.close(linger=0)
        pub_b.close(linger=0)
        ctx.term()
        await client.aclose()


@pytest.mark.asyncio
async def test_full_pipeline_handles_chained_parent_block_hash():
    """Worker emits two BlockStored events; second references the first
    via parent_block_hash. The router must chain through worker_hash →
    router_hash translation. Cache_view should end up with both."""
    port = _free_port()
    ep = f"tcp://127.0.0.1:{port}"

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(ep)

    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())
    w = _worker("w1", endpoint=ep, kv_block_size=4)
    policy.on_worker_added(w)

    try:
        encoder = msgspec.msgpack.Encoder()
        # First event: parent=None, one block [1..4]
        # Second event: parent=1001 (the just-stored block), one block [5..8]
        # The router has to translate worker hash 1001 → router hash on
        # the second event.
        batch = SglangKVEventBatch(
            ts=1.0,
            events=[
                SglangBlockStored(
                    block_hashes=[1001],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4],
                    block_size=4,
                    lora_id=None,
                    medium="device",
                ),
                SglangBlockStored(
                    block_hashes=[1002],
                    parent_block_hash=1001,
                    token_ids=[5, 6, 7, 8],
                    block_size=4,
                    lora_id=None,
                    medium="device",
                ),
            ],
        )
        payload = encoder.encode(batch)

        ok = await _publish_until_visible(
            pub,
            payload,
            predicate=lambda: len(client.cache_view("w1")) == 2,
            deadline_s=3.0,
        )
        assert ok, f"w1 view incomplete: {client.cache_view('w1')}"

        # Request for the same 8 tokens should match BOTH cached blocks.
        picked, blocks = policy.pick(
            [w], {"model": "test/m", "token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}
        )
        assert picked.worker.worker_id == "w1"

        # cache_hits is the prefix length — both blocks should hit.
        hits = policy._cache_hits(RouteTarget(w), {(EngineType.SGLANG, 4): blocks})
        assert hits == 2
    finally:
        pub.close(linger=0)
        ctx.term()
        await client.aclose()


@pytest.mark.asyncio
async def test_request_with_misaligned_tokens_gets_zero_hits():
    """Worker cached [1..8]; request asks for [9..16]. cache_hits = 0,
    so the cost reduces to overlap_weight * total + active. With one
    worker and overlap_weight=1, cost=2."""
    port = _free_port()
    ep = f"tcp://127.0.0.1:{port}"

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(ep)

    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())
    w = _worker("w1", endpoint=ep, kv_block_size=4)
    policy.on_worker_added(w)

    try:
        encoder = msgspec.msgpack.Encoder()
        batch = SglangKVEventBatch(
            ts=1.0,
            events=[
                SglangBlockStored(
                    block_hashes=[1001, 1002],
                    parent_block_hash=None,
                    token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    block_size=4,
                    lora_id=None,
                    medium="device",
                ),
            ],
        )
        payload = encoder.encode(batch)
        ok = await _publish_until_visible(
            pub,
            payload,
            predicate=lambda: len(client.cache_view("w1")) == 2,
            deadline_s=3.0,
        )
        assert ok

        picked, blocks = policy.pick(
            [w], {"model": "test/m", "token_ids": [9, 10, 11, 12, 13, 14, 15, 16]}
        )
        assert picked.worker.worker_id == "w1"
        # Hashes of [9..12] and [13..16] aren't in wA's view → 0 hits.
        assert policy._cache_hits(RouteTarget(w), {(EngineType.SGLANG, 4): blocks}) == 0
    finally:
        pub.close(linger=0)
        ctx.term()
        await client.aclose()
