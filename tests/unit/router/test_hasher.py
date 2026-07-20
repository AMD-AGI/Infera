###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for chained XXH3 hash primitives used by KV-aware routing."""

from __future__ import annotations

from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk, hash_request


def test_hash_chunk_is_deterministic():
    """Same parent + same tokens → same hash, every time."""
    h1 = hash_chunk(0, [101, 234, 567, 89])
    h2 = hash_chunk(0, [101, 234, 567, 89])
    assert h1 == h2


def test_hash_chunk_changes_with_parent():
    """Different parent → different hash (this is what 'chained' means)."""
    tokens = [101, 234, 567, 89]
    h1 = hash_chunk(0, tokens)
    h2 = hash_chunk(1, tokens)
    assert h1 != h2


def test_hash_chunk_changes_with_tokens():
    """Different tokens → different hash."""
    h1 = hash_chunk(0, [101, 234, 567, 89])
    h2 = hash_chunk(0, [101, 234, 567, 90])
    assert h1 != h2


def test_hash_request_chains_from_seed():
    """First block uses ROUTER_SEED as parent; subsequent blocks chain."""
    tokens = [101, 234, 567, 89, 12, 45, 78, 99]
    hashes = hash_request(tokens, block_size=4)

    assert len(hashes) == 2
    assert hashes[0] == hash_chunk(ROUTER_SEED, [101, 234, 567, 89])
    assert hashes[1] == hash_chunk(hashes[0], [12, 45, 78, 99])


def test_shared_prefix_yields_shared_prefix_hashes():
    """The whole point: two prompts with shared prefix get matching leading hashes."""
    a = [101, 234, 567, 89, 12, 45, 78, 99]
    b = [101, 234, 567, 89, 33, 44, 55, 66]
    ha = hash_request(a, block_size=4)
    hb = hash_request(b, block_size=4)

    assert ha[0] == hb[0]  # shared Block 0
    assert ha[1] != hb[1]  # diverging Block 1


def test_hash_request_discards_trailing_partial_block():
    """Engines only cache full blocks; the router must too."""
    tokens = [101, 234, 567, 89, 12]  # 5 tokens, block_size=4 → 1 full block
    hashes = hash_request(tokens, block_size=4)
    assert len(hashes) == 1


def test_hash_request_handles_empty():
    assert hash_request([], block_size=4) == []


def test_hash_request_handles_zero_block_size():
    """Defensive: block_size=0 from a misconfigured worker shouldn't crash."""
    assert hash_request([101, 234], block_size=0) == []


def test_hash_returns_64_bit_unsigned_int():
    """Used as a set element; should fit in int64 range comfortably."""
    h = hash_chunk(0, [101, 234, 567, 89])
    assert isinstance(h, int)
    assert 0 <= h < 2**64
