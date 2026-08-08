###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM emits one BlockStored per KV-cache group; only attention groups are indexable.

Measured on Kimi-K3 (3 KDA/Mamba groups + 1 MLA group, block_size 768, one
prefill chunk):

    kv_cache_spec_kind   group_idx   token_ids   block_hashes
    mamba                        0        3840              1
    mamba                        1        3840              1
    mamba                        2        3840              1
    mla_attention                3        3840              5

The Mamba groups run prefix caching in "align" mode, where all but one block per
step is a null block that is skipped when the hash list is built, while
``token_ids`` still spans the whole range. There is no field saying which chunk
the surviving hash covers, so the event cannot be indexed.

Indexing it anyway is not merely useless. vLLM's block hash does not mix in the
group id, so at equal block sizes a Mamba hash COLLIDES with an attention hash
and overwrites its entry in the engine-hash -> router-hash map; every later
attention event naming that hash as its parent then chains off the wrong node.
That is how 3 of 4 event streams silently destroyed the one usable stream, and
the symptom was ``cache_hits=0`` on every routing decision with all traffic
pinned to a single worker.

Upstream: vllm#44451 (open). NVIDIA Dynamo survives hybrid models only because
its `is_main_attention()` filter drops these groups before decoding.
"""

from __future__ import annotations

import logging

import pytest

from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.events import BlockStored
from infera.router.kv_event.hasher import hash_request

BS = 4


def _worker():
    return WorkerInfo(
        worker_id="w1:30000",
        url="http://w1:30000",
        model_name="m",
        engine=EngineType.VLLM,
        kv_events_endpoint="tcp://w1:5555",
        kv_block_size=BS,
    )


def _stored(*, blocks, spec_kind, group_idx=0, parent=None, first_hash=0, first_token=0):
    """An event whose lengths AGREE, so only the group filter can reject it.

    ``first_token`` matters: a follow-on chunk carries the NEXT tokens, not the
    same ones again, and hashing the wrong span makes the chain diverge from
    what a query reproduces.
    """
    return BlockStored(
        block_hashes=[bytes([first_hash + i]) for i in range(blocks)],
        parent_block_hash=parent,
        token_ids=list(range(first_token, first_token + blocks * BS)),
        block_size=BS,
        lora_id=None,
        group_idx=group_idx,
        kv_cache_spec_kind=spec_kind,
    )


async def _subscribed(client):
    client.on_worker_added(_worker())
    sub = client._subs["w1:30000"]
    for t in sub.tasks:
        t.cancel()
    return sub


@pytest.mark.parametrize("kind", ["full_attention", "mla_attention", "sink_full_attention"])
async def test_attention_groups_are_indexed(kind):
    """``mla_attention`` is not optional: Kimi-K3's attention layers are MLA, so
    a filter written as ``== "full_attention"`` drops 100% of its usable events
    — the same empty view the filter exists to prevent, reached from the other
    side."""
    client = KvEventClient()
    sub = await _subscribed(client)

    client._handle_event(sub, _stored(blocks=2, spec_kind=kind), rank=0)

    assert len(sub.map_for(0)) == 2
    assert len(sub.view_for(0)) == 2


@pytest.mark.parametrize("kind", ["mamba", "sliding_window", "encoder_only_attention"])
async def test_non_attention_groups_are_dropped(kind):
    client = KvEventClient()
    sub = await _subscribed(client)

    client._handle_event(sub, _stored(blocks=2, spec_kind=kind), rank=0)

    assert len(sub.map_for(0)) == 0
    assert len(sub.view_for(0)) == 0


async def test_absent_spec_kind_fails_open():
    """SGLang, and vLLM builds predating the field, send no kind at all. Those
    streams were indexed before this filter existed, so a closed default would
    silently switch kv-aware routing off for them — the same class of regression
    this filter is fixing."""
    client = KvEventClient()
    sub = await _subscribed(client)

    client._handle_event(sub, _stored(blocks=2, spec_kind=None), rank=0)

    assert len(sub.map_for(0)) == 2


async def test_a_later_mamba_event_cannot_poison_the_attention_chain():
    """The property the filter exists for, asserted through the query path.

    vLLM mixes no group id into the block hash, so at equal block sizes the
    Mamba and attention groups hand out the SAME hash for the same position.
    Order decides the damage: when the Mamba event arrives AFTER the attention
    event, it overwrites ``map[hash] -> router_hash`` with a hash computed over
    its own token span, and the next chunk -- which names that hash as its
    parent -- chains off the wrong node. The view then holds a block hash that
    no query can ever reproduce.

    Asserted the way routing actually asks: hash the full prompt and count how
    many of its blocks are in the view. Checking ``len(map)`` would not catch
    this, because the clobber replaces an entry rather than adding one.
    """
    client = KvEventClient()
    sub = await _subscribed(client)

    # attention group: blocks 0-1 of the prompt, tokens 0..7
    client._handle_event(sub, _stored(blocks=2, spec_kind="mla_attention", group_idx=3), rank=0)
    # mamba group: same hashes, DIFFERENT tokens, arriving second
    mamba = _stored(blocks=2, spec_kind="mamba", group_idx=0, first_token=100)
    client._handle_event(sub, mamba, rank=0)

    # the next chunk: block 2, tokens 8..11, parented on block 1
    client._handle_event(
        sub,
        _stored(
            blocks=1,
            spec_kind="mla_attention",
            group_idx=3,
            parent=bytes([1]),
            first_hash=2,
            first_token=2 * BS,
        ),
        rank=0,
    )

    # A query over the whole 3-block prompt must find all three.
    want = hash_request(list(range(3 * BS)), BS)
    assert len(want) == 3
    hits = sum(1 for h in want if h in sub.view_for(0))
    assert hits == 3, (
        f"{hits}/3 blocks visible: a later non-attention event overwrote the "
        "attention group's hash map and the chain continued from the wrong node"
    )


async def test_dropping_is_reported_once(caplog):
    """Silence here is how the original defect survived: a router that quietly
    indexes nothing looks exactly like one with a cold cache."""
    client = KvEventClient()
    sub = await _subscribed(client)

    with caplog.at_level(logging.INFO):
        for _ in range(3):
            client._handle_event(sub, _stored(blocks=2, spec_kind="mamba"), rank=0)

    hits = [r for r in caplog.records if "non-attention groups" in r.getMessage()]
    assert len(hits) == 1, "latched: three events must not produce three copies"
    assert "mamba" in hits[0].getMessage()
