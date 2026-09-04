###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A worker that registers no kv_block_size must not be subscribed, and events
whose token_ids and block_hashes disagree must not raise.

Both of these shipped broken and neither was caught, because every existing
router test builds events where the two agree — block_size=4 with four tokens
per hash. The failure needs the disagreement.

What it cost: Kimi-K3 is a hybrid Mamba model, so vLLM resolves the attention
block size to 768 after load while ``--block-size`` on the command line is still
None. infera registered the None, the router read it as ``or 1``, and
``_on_block_stored`` then derived its loop count from token_ids (768) and used it
to index block_hashes (1). Every event raised IndexError, the subscriber
reconnected forever, and 234 consecutive routing decisions reported
cache_hits=0 — kv-aware routing silently off, with a healthy-looking log.

These use the real ``BlockStored`` struct, not a stand-in. An earlier draft of
this file used a duck-typed class and every assertion passed vacuously:
``_handle_event`` dispatches on ``isinstance``, so the stand-in matched no branch
and the handler under test never ran.
"""

from __future__ import annotations

import logging

import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.events import BlockStored


def _worker(block_size, disagg_mode=DisaggMode.MIXED):
    return WorkerInfo(
        worker_id="w1:30000",
        url="http://w1:30000",
        model_name="m",
        engine=EngineType.VLLM,
        kv_events_endpoint="tcp://w1:5555",
        kv_block_size=block_size,
        disagg_mode=disagg_mode,
    )


def _stored(tokens, hashes, block_size, parent=None):
    return BlockStored(
        block_hashes=[bytes([i]) for i in range(hashes)],
        parent_block_hash=parent,
        token_ids=list(range(tokens)),
        block_size=block_size,
        lora_id=None,
    )


async def _subscribed(client, block_size):
    """Add a worker and hand back its subscription, tasks cancelled on teardown."""
    client.on_worker_added(_worker(block_size))
    sub = client._subs["w1:30000"]
    for t in sub.tasks:
        t.cancel()
    return sub


def test_worker_without_block_size_is_not_subscribed(caplog):
    """``or 1`` used to accept these, then hash per token against an engine
    paging per 768 — a view that can never match, reported as a healthy sub."""
    client = KvEventClient()
    with caplog.at_level(logging.ERROR):
        client.on_worker_added(_worker(None))

    assert client._subs == {}, "a worker with no block size must not be subscribed"
    assert any("kv_block_size" in r.getMessage() for r in caplog.records), (
        "refusing to subscribe has to say why, or it looks like the worker was missed"
    )

    # 0 is the same defect wearing a different value.
    client.on_worker_added(_worker(0))
    assert client._subs == {}


async def test_mismatched_event_indexes_blocks_but_not_hashes():
    """The real shape of the bug: 768 tokens, 1 block hash, subscriber at 1.

    The first fix stopped this raising IndexError. The second dropped the event
    whole, on the reasoning that a partial index binds an engine hash to a block
    it may not describe. That reasoning is right about the HASH and wrong about
    the BLOCK, and measuring it showed the cost: on a hybrid model emitting
    these shapes, dropping cut full-prefix hits from 22/32 to 18/32.

    The view entries are correct -- each is chained from a resolved parent over
    a contiguous token span, which is exactly what a query reproduces. Only the
    hash-to-block pairing is unknown, and only `map` uses it.

    So: index the blocks, withhold the hashes. A later event naming one of those
    hashes as its parent then misses and is dropped, which under-reports hits
    rather than chaining off the wrong node.
    """
    client = KvEventClient()
    sub = await _subscribed(client, 1)

    client._handle_event(sub, _stored(tokens=768, hashes=1, block_size=768), rank=0)

    assert len(sub.view_for(0)) == 768, "the blocks the span covers are indexable"
    assert len(sub.map_for(0)) == 0, "their hashes are not"


async def test_block_size_disagreement_is_reported_once(caplog):
    """The event carries the engine's block size; a mismatch names the fault
    directly instead of leaving it to be inferred from a zero hit rate."""
    client = KvEventClient()
    sub = await _subscribed(client, 1)

    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            client._handle_event(sub, _stored(tokens=768, hashes=1, block_size=768), rank=0)

    hits = [r for r in caplog.records if "paged at block_size" in r.getMessage()]
    assert len(hits) == 1, "latched: three events must not produce three copies"
    assert "768" in hits[0].getMessage()


async def test_agreeing_event_is_unaffected():
    """The bound must not truncate the normal case."""
    client = KvEventClient()
    sub = await _subscribed(client, 4)

    client._handle_event(sub, _stored(tokens=12, hashes=3, block_size=4), rank=0)

    assert len(sub.map_for(0)) == 3
    assert len(sub.view_for(0)) == 3


@pytest.mark.parametrize(
    "tokens,hashes,view,mapped",
    [
        (8, 2, 2, 2),  # agreeing: 2 hashes x block_size 4 == 8 tokens
        (8, 1, 2, 0),  # more tokens than hashes cover — the observed failure
        (4, 3, 1, 0),  # fewer tokens than hashes — the mirror case
        (0, 0, 0, 0),  # empty event
    ],
)
async def test_view_follows_the_tokens_and_the_map_follows_agreement(tokens, hashes, view, mapped):
    """The view is bounded by the token span, the map by the lengths agreeing.

    Both disagreement directions withhold the hashes: the direction of the error
    does not tell you which block the surviving hash belongs to.
    """
    client = KvEventClient()
    sub = await _subscribed(client, 4)

    client._handle_event(sub, _stored(tokens=tokens, hashes=hashes, block_size=4), rank=0)
    assert len(sub.view_for(0)) == view
    assert len(sub.map_for(0)) == mapped


@pytest.mark.parametrize("transport", ["zmq", "nats"])
def test_a_decode_leg_without_a_block_size_is_not_an_error(caplog, transport):
    """Refusing to track is right for every role; ERROR is right for only some.

    kv-aware routing never applies to the decode pool — the prefix hit is
    decided on the prefill side and disagg dispatch picks decode by load — so a
    decode leg arriving without a block size costs nothing. Logging it at ERROR
    on every PD startup is how a line that IS real for prefill gets filtered
    into the noise.

    Parametrised across both transports because they used to disagree, and the
    disagreement was invisible: the same worker got opposite treatment depending
    on how its events happened to reach the router. Asserting on one client
    would have passed throughout the window where the other was wrong.
    """
    client = _client_for(transport)
    with caplog.at_level(logging.DEBUG):
        client.on_worker_added(_worker(None, DisaggMode.DECODE))

    assert client._subs == {}, "still refused — only the severity changes"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "a decode leg with no block size is expected, not a fault"
    )
    assert any("decode" in r.getMessage() for r in caplog.records), (
        "silence would leave no way to tell this apart from a worker we missed"
    )


@pytest.mark.parametrize("transport", ["zmq", "nats"])
@pytest.mark.parametrize("mode", [DisaggMode.MIXED, DisaggMode.PREFILL])
def test_every_other_role_without_a_block_size_still_errors(caplog, transport, mode):
    """The half of the split that has to keep working. A prefill leg without a
    block size is an unresolved page size, and kv-aware is off for the pool that
    decides the hit."""
    client = _client_for(transport)
    with caplog.at_level(logging.DEBUG):
        client.on_worker_added(_worker(None, mode))

    assert client._subs == {}
    assert [r for r in caplog.records if r.levelno >= logging.ERROR], (
        f"{mode} with no block size is a fault and must stay loud"
    )


def _client_for(transport):
    if transport == "zmq":
        return KvEventClient()
    from infera.router.kv_event.nats_client import NatsKvEventClient

    return NatsKvEventClient(nats_url="nats://127.0.0.1:4222")
