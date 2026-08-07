###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end replay of the event shapes a real Kimi-K3 worker emits.

The per-field unit tests in ``test_kv_event_group_filter`` prove the filter
rejects what it should. They do not prove a request can HIT afterwards, which is
the property that actually matters and the one that stayed broken in production
after the filter went in: ``cache_hits=0`` on every routing decision.

Captured from a live worker (block_size 768, ``--max-num-batched-tokens 4096``,
so chunked prefill splits at ``4096 // 768 * 768 = 3840``):

    kv_cache_spec_kind   group_idx   token_ids   block_hashes
    mamba                        0        3840              1
    mamba                        1        3840              1
    mamba                        2        3840              1
    mla_attention                3        3840              5

Four groups per chunk, all at the same block size, and vLLM mixes no group id
into the hash -- so the Mamba groups' hashes collide with the attention group's.
"""

from __future__ import annotations

from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.events import BlockStored
from infera.router.kv_event.hasher import hash_request

BS = 768
CHUNK_BLOCKS = 5  # 3840 tokens / 768
CHUNK = BS * CHUNK_BLOCKS


def _worker():
    return WorkerInfo(
        worker_id="w1:30000",
        url="http://w1:30000",
        model_name="m",
        engine=EngineType.VLLM,
        kv_events_endpoint="tcp://w1:5555",
        kv_block_size=BS,
    )


async def _subscribed(client):
    client.on_worker_added(_worker())
    sub = client._subs["w1:30000"]
    for t in sub.tasks:
        t.cancel()
    return sub


def _chunk_events(tokens, first_block, parent):
    """One prefill chunk as four events, in the order vLLM emits them.

    The Mamba groups repeat the attention group's hashes because the engine
    derives them from the same token span with no group id mixed in. Each
    reports the whole span against a single hash -- align mode keeps exactly one
    real block per step and nulls the rest.
    """
    hashes = [bytes([first_block + i]) for i in range(len(tokens) // BS)]
    common = dict(
        parent_block_hash=parent,
        token_ids=list(tokens),
        block_size=BS,
        lora_id=None,
    )
    return [
        BlockStored(block_hashes=[hashes[-1]], group_idx=g, kv_cache_spec_kind="mamba", **common)
        for g in (0, 1, 2)
    ] + [
        BlockStored(block_hashes=hashes, group_idx=3, kv_cache_spec_kind="mla_attention", **common)
    ]


async def test_a_two_chunk_prefill_is_fully_hittable():
    """The property production needs: replay a prompt's events, then ask for the
    same prompt and get every block back.

    Two chunks, because a single one never exercises the parent map: chunk two
    names chunk one's last block as its parent, and that lookup is exactly what
    a colliding Mamba hash used to break.
    """
    client = KvEventClient()
    sub = await _subscribed(client)

    prompt = list(range(2 * CHUNK))
    for ev in _chunk_events(prompt[:CHUNK], first_block=0, parent=None):
        client._handle_event(sub, ev, rank=0)
    for ev in _chunk_events(
        prompt[CHUNK:], first_block=CHUNK_BLOCKS, parent=bytes([CHUNK_BLOCKS - 1])
    ):
        client._handle_event(sub, ev, rank=0)

    want = hash_request(prompt, BS)
    assert len(want) == 2 * CHUNK_BLOCKS
    hits = sum(1 for h in want if h in sub.view_for(0))
    assert hits == len(want), (
        f"{hits}/{len(want)} blocks visible after replaying a two-chunk prefill; "
        "a request for this prompt would route as if the cache were cold"
    )


async def test_a_partial_prefix_hits_its_prefix_only():
    """A shorter prompt sharing the first chunk must hit exactly that chunk --
    not more (which would be a false hit onto another prefix) and not less."""
    client = KvEventClient()
    sub = await _subscribed(client)

    prompt = list(range(2 * CHUNK))
    for ev in _chunk_events(prompt[:CHUNK], first_block=0, parent=None):
        client._handle_event(sub, ev, rank=0)
    for ev in _chunk_events(
        prompt[CHUNK:], first_block=CHUNK_BLOCKS, parent=bytes([CHUNK_BLOCKS - 1])
    ):
        client._handle_event(sub, ev, rank=0)

    shared = hash_request(prompt[:CHUNK], BS)
    assert sum(1 for h in shared if h in sub.view_for(0)) == CHUNK_BLOCKS

    divergent = hash_request(list(range(CHUNK)) + list(range(9_000, 9_000 + CHUNK)), BS)
    hits = sum(1 for h in divergent if h in sub.view_for(0))
    assert hits == CHUNK_BLOCKS, f"{hits} hits: a divergent tail must not match"


async def test_the_view_holds_only_the_attention_group():
    """Four groups arrive per chunk; only one is indexable, so the view must be
    one chunk's worth of blocks, not four."""
    client = KvEventClient()
    sub = await _subscribed(client)

    for ev in _chunk_events(list(range(CHUNK)), first_block=0, parent=None):
        client._handle_event(sub, ev, rank=0)

    assert len(sub.view_for(0)) == CHUNK_BLOCKS
    assert len(sub.map_for(0)) == CHUNK_BLOCKS
