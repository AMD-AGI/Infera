###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The MTP bigram token view in SGLang kv-events.

With EAGLE/MTP on, SGLang keys its radix tree on bigrams (``RadixKey.is_bigram``,
set from ``is_eagle``) and the kv-event emitter reports a block's tokens as the
overlapping pairs ``(t[i], t[i+1])`` rather than bare ints::

    sglang/srt/mem_cache/events.py
        is_bigram = node.key.is_bigram
        ...
        if is_bigram:
            page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]

Hashing the pairs as-is builds a cache view no query can ever match: the query
side chunks the FLAT token slice. The router therefore scores every worker zero
and kv-aware routing silently degrades to round-robin, with nothing in any log.

The property that matters is not merely "the view is non-empty under MTP" — it is
that the bigram view and the plain view produce the SAME hashes for the same
underlying tokens. That is what ``test_bigram_and_plain_views_agree`` asserts, and
it is the test that would fail on a flatten that took ``pair[1]`` instead.
"""

from __future__ import annotations

import pytest

from infera.router.kv_event import client as client_mod
from infera.router.kv_event.client import KvEventClient, WorkerSubscription
from infera.router.kv_event.events import SglangBlockStored
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk

# Imported by name rather than `from ... import _flat_tokens` on purpose: on
# pre-fix code the symbol does not exist, and a module-level ImportError would
# abort COLLECTION -- every test in the file erroring out, which proves nothing
# about the bug. This way the behavioural tests below run and genuinely fail.
_flat_tokens = getattr(client_mod, "_flat_tokens", None)
_needs_fix = pytest.mark.skipif(
    _flat_tokens is None, reason="_flat_tokens is the fix under test"
)


def _bigrams(tokens: list[int]) -> list[tuple[int, int]]:
    """The pairs SGLang emits for ``tokens``: ``(t[i], t[i+1])`` per position.

    The engine holds ``N+1`` raw tokens for ``N`` bigram positions, so the pair
    at the last position reaches one token past the block. Mirrored here.
    """
    return [(tokens[i], tokens[i + 1] if i + 1 < len(tokens) else 0) for i in range(len(tokens))]


# ----------------------------------------------------------------------
# _flat_tokens
# ----------------------------------------------------------------------


@_needs_fix
def test_flat_tokens_unwraps_pairs():
    assert _flat_tokens([(1, 2), (2, 3), (3, 4)]) == [1, 2, 3]


@_needs_fix
def test_flat_tokens_accepts_lists_not_only_tuples():
    """msgpack decodes a 2-element sequence as a list, not a tuple."""
    assert _flat_tokens([[1, 2], [2, 3]]) == [1, 2]


@_needs_fix
def test_flat_tokens_leaves_plain_ints_alone():
    assert _flat_tokens([1, 2, 3]) == [1, 2, 3]


@_needs_fix
def test_flat_tokens_handles_empty():
    assert _flat_tokens([]) == []


# ----------------------------------------------------------------------
# _handle_event end to end
# ----------------------------------------------------------------------


def test_bigram_event_populates_the_view():
    """Fails on the pre-fix code: the pairs hash to something no query matches.

    Pre-fix this does not raise — it produces a view of the WRONG hashes, which is
    precisely why the bug was invisible in production. So assert the exact
    expected hash, not just non-emptiness.
    """
    client = KvEventClient()
    sub = WorkerSubscription(worker_id="w1", endpoint="tcp://x:1", block_size=4)

    client._handle_event(
        sub,
        SglangBlockStored(
            block_hashes=[111],
            parent_block_hash=None,
            token_ids=_bigrams([1, 2, 3, 4]),
            block_size=4,
            lora_id=None,
        ),
    )

    expected = hash_chunk(ROUTER_SEED, [1, 2, 3, 4])
    assert sub.view_for(None) == {expected}
    assert sub.map_for(None) == {111: expected}


def test_bigram_and_plain_views_agree():
    """The same tokens must chain to the same hashes either way.

    This is the real invariant. A flatten that picked ``pair[1]`` would still give
    a non-empty view and still pass a "kv-aware works under MTP" smoke test, while
    keying every block one token off.
    """
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]

    plain = KvEventClient()
    plain_sub = WorkerSubscription(worker_id="p", endpoint="tcp://x:1", block_size=4)
    plain._handle_event(
        plain_sub,
        SglangBlockStored(
            block_hashes=[1, 2],
            parent_block_hash=None,
            token_ids=list(tokens),
            block_size=4,
            lora_id=None,
        ),
    )

    bigram = KvEventClient()
    bigram_sub = WorkerSubscription(worker_id="b", endpoint="tcp://x:1", block_size=4)
    bigram._handle_event(
        bigram_sub,
        SglangBlockStored(
            block_hashes=[1, 2],
            parent_block_hash=None,
            token_ids=_bigrams(tokens),
            block_size=4,
            lora_id=None,
        ),
    )

    assert bigram_sub.view_for(None) == plain_sub.view_for(None)
    assert bigram_sub.map_for(None) == plain_sub.map_for(None)


def test_bigram_event_decodes_over_the_wire():
    """The msgspec schema must accept pairs, else every event fails to decode.

    ``token_ids`` is annotated ``list[int | tuple[int, int]]``; a schema still
    declaring ``list[int]`` raises here rather than degrading quietly.
    """
    import msgspec
    from msgspec.msgpack import Decoder

    from infera.router.kv_event.events import SglangKVEventBatch

    ev = SglangBlockStored(
        block_hashes=[111],
        parent_block_hash=None,
        token_ids=_bigrams([1, 2, 3, 4]),
        block_size=4,
        lora_id=None,
    )
    raw = msgspec.msgpack.encode(SglangKVEventBatch(ts=0.0, events=[ev]))
    batch = Decoder(type=SglangKVEventBatch).decode(raw)

    (decoded,) = batch.events
    # One pair per position, so flattening recovers all four first elements.
    assert _flat_tokens(decoded.token_ids) == [1, 2, 3, 4]
