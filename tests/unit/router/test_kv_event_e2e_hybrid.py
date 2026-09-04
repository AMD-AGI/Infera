###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Two workers, real ZMQ, a hybrid model's event stream — does routing follow the cache?

``test_kv_event_e2e.py`` covers the same pipeline for SGLang. This is the vLLM
hybrid case, which fails differently and was found in production rather than by
a test: kv-aware routing reported ``cache_hits=0`` on every decision and pinned
all traffic to one worker, leaving the other node's GPUs idle. That is not a
degraded hit rate, it is half the fleet.

The shapes below are what a live Kimi-K3 worker publishes, captured off the
wire (block_size 768, ``--max-num-batched-tokens 4096`` so chunked prefill
splits at ``4096 // 768 * 768 = 3840``):

    kv_cache_spec_kind   group_idx   token_ids   block_hashes
    mamba                        0        3840              1
    mamba                        1        3840              1
    mamba                        2        3840              1
    mla_attention                3        3840              5

Three of the four groups are unusable: prefix caching runs in "align" mode on
the KDA layers, so all but one block per step is a null block skipped when the
hash list is built, while ``token_ids`` still spans everything. Nothing says
which chunk the surviving hash covers.

They are also actively destructive. vLLM mixes no group id into the block hash,
so at equal block sizes a Mamba hash COLLIDES with an attention hash and
overwrites its entry in the engine-hash -> router-hash map; the next chunk then
resolves its parent to the wrong node. Three streams silently destroyed the
fourth, which is why the fix is a filter and not a lenient decoder.

Uses real ZMQ rather than calling ``_handle_event`` directly, so the msgspec
schema is exercised: the two fields the filter depends on, ``group_idx`` and
``kv_cache_spec_kind``, were on the wire all along and were being discarded
because the struct did not declare them. A test that constructs events in
Python cannot catch that.
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
from infera.router.policy.kv_event_aware import KvEventAwarePolicy
from infera.router.policy.target import RouteTarget

_TOPIC = b"kv-events"
BS = 768
CHUNK_BLOCKS = 5
CHUNK = BS * CHUNK_BLOCKS


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(worker_id: str, endpoint: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url=f"http://{worker_id}",
        model_name="test/m",
        engine=EngineType.VLLM,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=endpoint,
        kv_block_size=BS,
    )


class _IdentityHasher:
    """Treat ``body["token_ids"]`` as already tokenized, so request/worker
    alignment is controlled by the test rather than by a tokenizer."""

    # BlockHasher's gate for `spawn_probe`; these doubles always render.
    def can_render(self, model_id, engine=None) -> bool:
        return True

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        from infera.router.kv_event.hasher import hash_request

        return hash_request(body.get("token_ids", []), block_size)


def _chunk_payload(
    tokens: list[int], first_block: int, parent: bytes | None, *, mamba_last: bool = False
) -> bytes:
    """One prefill chunk, encoded exactly as vLLM puts it on the wire.

    Built as plain dicts, not our own structs, so the test does not inherit the
    schema it is meant to be checking.

    ``mamba_last`` exists because ORDER DECIDED CORRECTNESS before the fix, and
    an earlier draft of this test missed the bug entirely by picking the lucky
    order. With the attention event last, its five correct hashes overwrite
    whatever the Mamba events wrote and the view comes out right by accident.
    Put the Mamba events last and they clobber the attention group's map, so
    the next chunk resolves its parent to the wrong node.

    vLLM does not promise either order. A router whose cache view depends on it
    is broken whichever way the engine happens to emit today.
    """
    hashes = [bytes([first_block + i]) for i in range(len(tokens) // BS)]
    base = {
        "type": "BlockStored",
        "parent_block_hash": parent,
        "token_ids": tokens,
        "block_size": BS,
        "lora_id": None,
    }
    mamba = [
        {**base, "block_hashes": [hashes[-1]], "group_idx": g, "kv_cache_spec_kind": "mamba"}
        for g in (0, 1, 2)
    ]
    attn = {**base, "block_hashes": hashes, "group_idx": 3, "kv_cache_spec_kind": "mla_attention"}
    events = [attn, *mamba] if mamba_last else [*mamba, attn]
    return msgspec.msgpack.encode([0.0, events, None])


async def _publish_until(
    pub: Any, payloads: list[bytes], predicate, *, deadline_s: float = 5.0
) -> bool:
    """PUB/SUB drops messages sent before the subscriber attaches, so resend
    until the effect is visible or the deadline passes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while loop.time() < deadline:
        for p in payloads:
            pub.send_multipart([_TOPIC, p])
        await asyncio.sleep(0.05)
        if predicate():
            return True
    return False


@pytest.mark.parametrize("mamba_last", [False, True], ids=["attn-last", "mamba-last"])
@pytest.mark.asyncio
async def test_routing_follows_the_cache_across_two_workers(mamba_last):
    """The production scenario, end to end, under BOTH intra-batch orders.

    Worker A serves a prompt; its events go out. A request for the SAME prompt
    must then pick A over an idle B.

    Both orders are run because only one of them fails without the fix. With
    the attention event last its correct hashes overwrite the Mamba groups'
    and everything works by luck; with the Mamba events last they overwrite the
    attention group's map and the second chunk chains off the wrong parent. The
    fix makes the outcome independent of an order vLLM never promised.
    """
    ctx = zmq.Context.instance()
    port_a, port_b = _free_port(), _free_port()
    pub_a = ctx.socket(zmq.PUB)
    pub_a.bind(f"tcp://127.0.0.1:{port_a}")
    pub_b = ctx.socket(zmq.PUB)
    pub_b.bind(f"tcp://127.0.0.1:{port_b}")

    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())
    wa = _worker("a:30000", f"tcp://127.0.0.1:{port_a}")
    wb = _worker("b:30000", f"tcp://127.0.0.1:{port_b}")
    policy.on_worker_added(wa)
    policy.on_worker_added(wb)

    prompt = list(range(2 * CHUNK))
    payloads = [
        _chunk_payload(prompt[:CHUNK], 0, None, mamba_last=mamba_last),
        _chunk_payload(
            prompt[CHUNK:], CHUNK_BLOCKS, bytes([CHUNK_BLOCKS - 1]), mamba_last=mamba_last
        ),
    ]
    try:
        ok = await _publish_until(
            pub_a,
            payloads,
            lambda: len(client._subs["a:30000"].view_for(None)) >= 2 * CHUNK_BLOCKS,
        )
        assert ok, (
            "worker A's view never filled: the attention group's events are not "
            "being indexed at all"
        )
        assert len(client._subs["b:30000"].view_for(None)) == 0, "B was never fed"

        # The assertion that matters. View SIZE is not it: a poisoned chain
        # still produces ten entries, five of them hashed from the wrong parent,
        # and A still beats an empty B -- so "the right worker won" passes while
        # half the prompt is invisible. Count what actually matches.
        from infera.router.kv_event.hasher import hash_request

        want = hash_request(prompt, BS)
        view = client._subs["a:30000"].view_for(None)
        matched = sum(1 for h in want if h in view)
        assert matched == 2 * CHUNK_BLOCKS, (
            f"{matched}/{2 * CHUNK_BLOCKS} blocks of the prompt are visible on the "
            "worker that just served it; the chain continued from a clobbered parent"
        )

        target, blocks = policy.pick([wa, wb], {"model": "test/m", "token_ids": prompt})
        assert isinstance(target, RouteTarget)
        assert target.worker.worker_id == "a:30000"
        assert len(blocks) == 2 * CHUNK_BLOCKS
    finally:
        await client.aclose()
        pub_a.close(linger=0)
        pub_b.close(linger=0)


@pytest.mark.asyncio
async def test_the_view_is_one_groups_worth_not_four():
    """Four events arrive per chunk and three are unusable. If they were all
    indexed the view would be inflated with blocks hashed from a Mamba group's
    span -- which is how the collision corrupts the map."""
    ctx = zmq.Context.instance()
    port = _free_port()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")

    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())
    policy.on_worker_added(_worker("a:30000", f"tcp://127.0.0.1:{port}"))
    try:
        payload = _chunk_payload(list(range(CHUNK)), 0, None)
        ok = await _publish_until(
            pub,
            [payload],
            lambda: len(client._subs["a:30000"].view_for(None)) >= CHUNK_BLOCKS,
        )
        assert ok, "the attention group's chunk never landed"
        # Give the other three groups every chance to be indexed too.
        for _ in range(5):
            pub.send_multipart([_TOPIC, payload])
            await asyncio.sleep(0.05)

        sub = client._subs["a:30000"]
        assert len(sub.view_for(None)) == CHUNK_BLOCKS
        assert len(sub.map_for(None)) == CHUNK_BLOCKS
    finally:
        await client.aclose()
        pub.close(linger=0)
