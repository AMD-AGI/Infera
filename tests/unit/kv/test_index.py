###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from infera.kv.hashing import hash_token_blocks
from infera.kv.index import KVIndex
from infera.kv.types import BlockKey, OverlapBlocks, Tier


def _chain(tokens: list[int], block_size: int = 4) -> list[BlockKey]:
    """Real hash chain from tokens. Tests use this so block_hash and
    sequence_hash values are correct relative to each other (the index
    sanity-checks the relationship)."""
    return hash_token_blocks(tokens, block_size=block_size)


def _store_chain(
    index: KVIndex,
    publisher_id: str,
    model: str,
    compat_key: str,
    chain: list[BlockKey],
    tier: Tier,
    now_ms: int = 1000,
) -> None:
    for block in chain:
        index.apply_stored(
            publisher_id=publisher_id,
            model=model,
            compat_key=compat_key,
            block=block,
            tier=tier,
            now_ms=now_ms,
        )


# ----------------------------------------------------------------------
# Basic apply / find
# ----------------------------------------------------------------------


def test_apply_stored_then_find_full_overlap() -> None:
    idx = KVIndex()
    chain = _chain(list(range(16)))  # 4 blocks
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    assert matches["w1"] == OverlapBlocks(device=4)


def test_find_unknown_publisher_returns_empty() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    matches = idx.find_matches(
        model="m", compat_key="ck", chain=chain, candidates=["unknown-worker"]
    )
    assert matches["unknown-worker"] == OverlapBlocks()


def test_find_empty_chain() -> None:
    idx = KVIndex()
    matches = idx.find_matches(model="m", compat_key="ck", chain=[], candidates=["w1"])
    assert matches["w1"] == OverlapBlocks()


def test_apply_stored_idempotent() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    assert matches["w1"] == OverlapBlocks(device=2)
    # Stats: 2 nodes, all confirmed.
    s = idx.stats()
    assert s.nodes == 2
    assert s.confirmed_nodes == 2
    assert s.optimistic_nodes == 0


# ----------------------------------------------------------------------
# Diverging chains
# ----------------------------------------------------------------------


def test_diverging_chain_counts_only_common_prefix() -> None:
    idx = KVIndex()
    base_tokens = list(range(16))  # 4 blocks
    _store_chain(idx, "w1", "m", "ck", _chain(base_tokens), Tier.DEVICE)

    # Query with a chain that shares the first 2 blocks then diverges.
    divergent_tokens = list(range(8)) + [999, 998, 997, 996] + [10, 11, 12, 13]
    matches = idx.find_matches(
        model="m",
        compat_key="ck",
        chain=_chain(divergent_tokens),
        candidates=["w1"],
    )
    # Only the first 2 blocks (tokens 0..8) match.
    assert matches["w1"].device == 2
    # Nothing on host/disk/fabric.
    assert matches["w1"].host == 0


# ----------------------------------------------------------------------
# Multi-tier
# ----------------------------------------------------------------------


def test_block_on_multiple_tiers_counts_per_tier() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))  # 2 blocks
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m", "ck", chain, Tier.HOST)
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    assert matches["w1"].device == 2
    assert matches["w1"].host == 2


# ----------------------------------------------------------------------
# Remove
# ----------------------------------------------------------------------


def test_apply_removed_drops_tier() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m", "ck", chain, Tier.HOST)
    # Remove only the device tier from the first block.
    idx.apply_removed(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        sequence_hash=chain[0].sequence_hash,
        tier=Tier.DEVICE,
    )
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    # Block 0 still has host, block 1 still has device and host.
    assert matches["w1"].device == 1  # only block 1
    assert matches["w1"].host == 2  # both


def test_apply_removed_full_tier_unlinks_leaf() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))  # 2 blocks
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    # Remove device from block 1 (leaf).
    idx.apply_removed(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        sequence_hash=chain[1].sequence_hash,
        tier=Tier.DEVICE,
    )
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    # Only block 0 remains.
    assert matches["w1"].device == 1
    # _by_seq shrunk.
    assert idx.stats().nodes == 1


def test_apply_removed_idempotent() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    # Remove twice; second is a no-op.
    for _ in range(2):
        idx.apply_removed(
            publisher_id="w1",
            model="m",
            compat_key="ck",
            sequence_hash=chain[0].sequence_hash,
            tier=Tier.DEVICE,
        )
    # No exception. Block 1 still there (we only removed block 0's tier).
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    # Block 0 lost device; but we walk by chain so we stop at block 0 mismatch?
    # Actually block 0's node still exists (it has children), but tiers is empty
    # → device count = 0 for it; block 1 has device, but we may not reach it
    # because walking from root, the first child (block 0) has empty tiers.
    # Walk still proceeds (we stop on missing child, not on empty tiers).
    assert matches["w1"].device == 1  # only block 1 contributes


def test_apply_removed_unknown_sequence_hash_is_noop() -> None:
    idx = KVIndex()
    idx.apply_removed(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        sequence_hash=0xDEADBEEFDEADBEEF,
        tier=Tier.DEVICE,
    )  # Doesn't raise.


# ----------------------------------------------------------------------
# Cleared
# ----------------------------------------------------------------------


def test_apply_cleared_all() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m1", "ck1", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m2", "ck2", chain, Tier.DEVICE)
    idx.apply_cleared(publisher_id="w1", scope="all")
    matches1 = idx.find_matches(model="m1", compat_key="ck1", chain=chain, candidates=["w1"])
    matches2 = idx.find_matches(model="m2", compat_key="ck2", chain=chain, candidates=["w1"])
    assert matches1["w1"] == OverlapBlocks()
    assert matches2["w1"] == OverlapBlocks()
    assert idx.stats().nodes == 0


def test_apply_cleared_model_scope_preserves_others() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m1", "ck1", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m2", "ck1", chain, Tier.DEVICE)
    idx.apply_cleared(publisher_id="w1", scope="model:m1")
    matches1 = idx.find_matches(model="m1", compat_key="ck1", chain=chain, candidates=["w1"])
    matches2 = idx.find_matches(model="m2", compat_key="ck1", chain=chain, candidates=["w1"])
    assert matches1["w1"].device == 0
    assert matches2["w1"].device == 2  # untouched


def test_apply_cleared_compat_key_scope() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ckA", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m", "ckB", chain, Tier.DEVICE)
    idx.apply_cleared(publisher_id="w1", scope="compat_key:ckA")
    a = idx.find_matches(model="m", compat_key="ckA", chain=chain, candidates=["w1"])
    b = idx.find_matches(model="m", compat_key="ckB", chain=chain, candidates=["w1"])
    assert a["w1"].device == 0
    assert b["w1"].device == 2


def test_apply_cleared_tier_scope() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m", "ck", chain, Tier.HOST)
    idx.apply_cleared(publisher_id="w1", scope="tier:device")
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
    assert matches["w1"].device == 0
    assert matches["w1"].host == 2


def test_apply_cleared_unknown_tier_is_noop() -> None:
    idx = KVIndex()
    idx.apply_cleared(publisher_id="w1", scope="tier:notreal")  # No exception.


# ----------------------------------------------------------------------
# Drop publisher
# ----------------------------------------------------------------------


def test_drop_publisher_removes_all_for_publisher() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE)
    _store_chain(idx, "w2", "m", "ck", chain, Tier.DEVICE)
    idx.drop_publisher("w1")
    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1", "w2"])
    assert matches["w1"] == OverlapBlocks()
    assert matches["w2"].device == 2


# ----------------------------------------------------------------------
# Optimistic recording
# ----------------------------------------------------------------------


def test_optimistic_record_visible_in_find() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    idx.optimistically_record(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        chain=chain,
        tier=Tier.DEVICE,
        ttl_s=2.0,
        now_ms=1000,
    )
    matches = idx.find_matches(
        model="m", compat_key="ck", chain=chain, candidates=["w1"], now_ms=1500
    )
    assert matches["w1"].device == 2
    # Marked as optimistic in stats.
    s = idx.stats()
    assert s.optimistic_nodes == 2


def test_optimistic_record_evicted_after_ttl() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    idx.optimistically_record(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        chain=chain,
        tier=Tier.DEVICE,
        ttl_s=2.0,
        now_ms=1000,
    )
    # Step past the TTL deadline and sweep.
    removed = idx.evict_expired(now_ms=1000 + 2001)
    assert removed == 2
    matches = idx.find_matches(
        model="m", compat_key="ck", chain=chain, candidates=["w1"], now_ms=4000
    )
    assert matches["w1"] == OverlapBlocks()


def test_optimistic_record_confirmed_by_real_event() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    idx.optimistically_record(
        publisher_id="w1",
        model="m",
        compat_key="ck",
        chain=chain,
        tier=Tier.DEVICE,
        ttl_s=2.0,
        now_ms=1000,
    )
    # Real event arrives before TTL — should clear optimistic mark.
    for block in chain:
        idx.apply_stored(
            publisher_id="w1",
            model="m",
            compat_key="ck",
            block=block,
            tier=Tier.DEVICE,
            now_ms=1100,
        )
    # Now sweep at a time *past* the original optimistic TTL.
    removed = idx.evict_expired(now_ms=1000 + 5000)
    # Default TTL is 600 s, so confirmed nodes should not be evicted at t=6 s.
    assert removed == 0
    matches = idx.find_matches(
        model="m", compat_key="ck", chain=chain, candidates=["w1"], now_ms=6000
    )
    assert matches["w1"].device == 2


# ----------------------------------------------------------------------
# TTL on confirmed nodes
# ----------------------------------------------------------------------


def test_confirmed_node_evicted_after_default_ttl() -> None:
    idx = KVIndex(default_ttl_s=1.0)  # very short TTL for the test
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE, now_ms=1000)
    removed = idx.evict_expired(now_ms=1000 + 2000)
    assert removed == 2


def test_confirmed_node_not_evicted_within_ttl() -> None:
    idx = KVIndex(default_ttl_s=10.0)
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE, now_ms=1000)
    removed = idx.evict_expired(now_ms=1000 + 500)
    assert removed == 0


def test_confirmed_node_refreshed_by_hit_survives_ttl() -> None:
    """Eviction now uses last_access_ts_ms, not
    inserted_ts_ms. A popular block that's hit by `find_matches` gets
    refreshed and must survive the TTL window from its original
    insert."""
    idx = KVIndex(default_ttl_s=2.0)
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ck", chain, Tier.DEVICE, now_ms=1000)
    # Just before old TTL would expire, hit the node — refreshes
    # last_access_ts_ms to 2500.
    idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"], now_ms=2500)
    # Now-time = 1000 + 3000 = 4000. inserted_ts_ms was 1000 (3s old →
    # would expire under old policy). last_access_ts_ms was refreshed
    # to 2500 (1.5s old → survives under new policy).
    removed = idx.evict_expired(now_ms=1000 + 3000)
    assert removed == 0
    # Time-since-access now exceeds TTL → eviction.
    removed = idx.evict_expired(now_ms=2500 + 3000)
    assert removed == 2


# ----------------------------------------------------------------------
# Multiple candidates / publisher isolation
# ----------------------------------------------------------------------


def test_independent_trees_per_publisher() -> None:
    idx = KVIndex()
    chain = _chain(list(range(16)))
    _store_chain(idx, "w1", "m", "ck", chain[:2], Tier.DEVICE)  # only first 2 blocks
    _store_chain(idx, "w2", "m", "ck", chain, Tier.DEVICE)  # all 4

    matches = idx.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1", "w2"])
    assert matches["w1"].device == 2
    assert matches["w2"].device == 4


def test_publisher_isolation_across_compat_key() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m", "ckA", chain, Tier.DEVICE)
    # Query for the same publisher and model but different compat_key — no hit.
    matches = idx.find_matches(model="m", compat_key="ckB", chain=chain, candidates=["w1"])
    assert matches["w1"] == OverlapBlocks()


# ----------------------------------------------------------------------
# Stats sanity
# ----------------------------------------------------------------------


def test_stats_reports_counts() -> None:
    idx = KVIndex()
    chain = _chain(list(range(8)))
    _store_chain(idx, "w1", "m1", "ck1", chain, Tier.DEVICE)
    _store_chain(idx, "w2", "m1", "ck1", chain, Tier.DEVICE)
    _store_chain(idx, "w1", "m2", "ck2", chain, Tier.DEVICE)
    s = idx.stats()
    assert s.publishers == 2  # w1, w2
    assert s.models == 2  # m1, m2
    assert s.compat_keys == 2  # ck1, ck2
    assert s.nodes == 6  # 3 trees × 2 blocks
    assert s.confirmed_nodes == 6
    assert s.optimistic_nodes == 0
