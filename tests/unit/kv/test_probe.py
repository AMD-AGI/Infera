###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the engine-agnostic KvEventProbe and the SGLang adapter.

The SGLang adapter is exercised against a `MockRadixCache` — a tiny
shim mimicking SGLang's `RadixCache.insert / evict / reset` and
`TreeNode.key / parent` shape — so the tests can run without SGLang
installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from infera.engine.sglang.kv_probe import (
    _walk_full_prefix_tokens,
    attach_to_radix_cache,
    detach_from_radix_cache,
)
from infera.kv.hashing import hash_token_blocks
from infera.kv.index import KVIndex
from infera.kv.probe import KvEventProbe
from infera.kv.publisher import KvEventPublisher
from infera.kv.snapshot import SnapshotProducer
from infera.kv.subscriber import KvEventSubscriber
from infera.kv.types import OverlapBlocks, Tier
from infera.kv.wire import EventType
from infera.kv.writer import KvIndexWriter

# Each test gets its own inproc:// endpoint to avoid context leakage.
_endpoint_counter = 0


def _next_endpoint() -> str:
    global _endpoint_counter
    _endpoint_counter += 1
    return f"inproc://infera-probe-test-{_endpoint_counter}"


# ----------------------------------------------------------------------
# Tiny in-memory publisher stub so we can unit-test the probe without ZMQ
# ----------------------------------------------------------------------


class _StubPublisher:
    """Captures emitted events in a list. Same interface as KvEventPublisher.emit."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, Any]] = []

    def emit(self, *, model: str, compat_key: str, event) -> None:
        self.events.append((model, compat_key, event))


def _make_probe(publisher=None, snapshot=None, index_block_size: int = 4, **kwargs) -> KvEventProbe:
    return KvEventProbe(
        publisher=publisher or _StubPublisher(),  # type: ignore[arg-type]
        snapshot_producer=snapshot,
        model_name=kwargs.pop("model_name", "m"),
        compat_key=kwargs.pop("compat_key", "ck"),
        index_block_size=index_block_size,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Engine-agnostic KvEventProbe
# ----------------------------------------------------------------------


def test_probe_rejects_zero_index_block_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        KvEventProbe(
            publisher=_StubPublisher(),  # type: ignore[arg-type]
            snapshot_producer=None,
            model_name="m",
            compat_key="ck",
            index_block_size=0,
        )


def test_probe_emits_stored_per_block() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    # 8 tokens / block_size=4 → 2 blocks → 2 STORED events.
    assert len(pub.events) == 2
    assert all(e.type == EventType.STORED for _, _, e in pub.events)
    assert probe.stored_emitted == 2


def test_probe_dedup_repeat_inserts() -> None:
    """Same prefix inserted twice → only one wave of STORED events."""
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    n_after_first = len(pub.events)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    assert len(pub.events) == n_after_first  # no new events


def test_probe_emits_new_blocks_when_prefix_extends() -> None:
    """Insert tokens [0..8), then [0..16). Second call emits only the new
    blocks (8..16)."""
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    assert len(pub.events) == 2
    probe.on_node_inserted(full_prefix_tokens=list(range(16)))
    # Should emit the two NEW blocks (8..12 and 12..16); the first two are dedup'd.
    assert len(pub.events) == 4


def test_probe_partial_block_not_emitted() -> None:
    """6 tokens / block_size=4 → only 1 full block."""
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(6)))
    assert len(pub.events) == 1


def test_probe_evict_emits_removed_for_last_block() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    stored_count = len(pub.events)
    probe.on_node_evicted(full_prefix_tokens=list(range(8)))
    # One additional REMOVED event for the tail block.
    assert len(pub.events) == stored_count + 1
    removed = pub.events[-1][2]
    assert removed.type == EventType.REMOVED


def test_probe_evict_idempotent_when_not_emitted() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    # Evict without prior insert — should be a no-op (nothing in _emitted).
    probe.on_node_evicted(full_prefix_tokens=list(range(8)))
    assert pub.events == []
    assert probe.removed_emitted == 0


def test_probe_on_clear_drops_emitted_set() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    probe.on_clear()
    # One cleared event emitted.
    cleared = [e for _, _, e in pub.events if e.type == EventType.CLEARED]
    assert len(cleared) == 1
    # Dedup set cleared; subsequent insert re-emits.
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    stored_after = [e for _, _, e in pub.events if e.type == EventType.STORED]
    # 2 initial + 2 after clear.
    assert len(stored_after) == 4


def test_probe_uses_configured_tier_and_role() -> None:
    pub = _StubPublisher()
    probe = _make_probe(
        publisher=pub,
        index_block_size=4,
        tier=Tier.HOST,
        role="sliding",
        group_idx=2,
    )
    probe.on_node_inserted(full_prefix_tokens=list(range(4)))
    e = pub.events[0][2]
    assert e.tier == "host"
    assert e.role == "sliding"
    assert e.group_idx == 2


def test_probe_pool_fields_propagate() -> None:
    """For events emitted by a pool daemon, pool_id and pool_type carry through."""
    pub = _StubPublisher()
    probe = _make_probe(
        publisher=pub,
        index_block_size=4,
        pool_id="infera-kvd-node-1",
        pool_type="infera-kvd",
    )
    probe.on_node_inserted(full_prefix_tokens=list(range(4)))
    e = pub.events[0][2]
    assert e.pool_id == "infera-kvd-node-1"
    assert e.pool_type == "infera-kvd"


def test_probe_writes_to_snapshot_producer() -> None:
    """When a snapshot producer is wired in, the probe mirrors events into it."""
    producer = SnapshotProducer(publisher_id="w1", publisher_type="worker", index_block_size=4)
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, snapshot=producer, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    snap = producer.snapshot(model="m", compat_key="ck")
    assert len(snap.blocks) == 2


def test_probe_metrics_counts() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    probe.on_node_inserted(full_prefix_tokens=list(range(8)))
    probe.on_node_evicted(full_prefix_tokens=list(range(8)))
    probe.on_clear()
    assert probe.stored_emitted == 2
    assert probe.removed_emitted == 1
    assert probe.cleared_emitted == 1
    assert probe.callbacks_received == 3


# ----------------------------------------------------------------------
# End-to-end: probe → ZMQ publisher → subscriber → writer → KVIndex
# ----------------------------------------------------------------------


async def test_probe_end_to_end_to_index() -> None:
    """Real ZMQ + writer + KVIndex; probe drives the whole pipeline."""
    endpoint = _next_endpoint()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    index = KVIndex()
    pub = KvEventPublisher(
        bind_endpoint=endpoint,
        publisher_id="w1",
        publisher_type="worker",
        index_block_size=4,
        batch_max_events=10,
        batch_max_ms=30,
    )
    sub = KvEventSubscriber(endpoint=endpoint, output_queue=queue)
    writer = KvIndexWriter(index=index, queue=queue)
    probe = KvEventProbe(
        publisher=pub,
        snapshot_producer=None,
        model_name="m",
        compat_key="ck",
        index_block_size=4,
    )
    await pub.start()
    await sub.start()
    await writer.start()
    try:
        await asyncio.sleep(0.05)
        probe.on_node_inserted(full_prefix_tokens=list(range(16)))  # 4 blocks
        await asyncio.sleep(0.3)
        chain = hash_token_blocks(list(range(16)), block_size=4)
        matches = index.find_matches(model="m", compat_key="ck", chain=chain, candidates=["w1"])
        assert matches["w1"] == OverlapBlocks(device=4)
    finally:
        await writer.stop()
        await sub.stop()
        await pub.stop()


# ----------------------------------------------------------------------
# SGLang adapter against a MockRadixCache
# ----------------------------------------------------------------------


@dataclass
class _MockTreeNode:
    """Stand-in for SGLang's TreeNode. Has `key` (token segment) and
    `parent` (chain to root). The probe walks parent.parent... to compute
    the full prefix."""

    key: list[int]
    parent: _MockTreeNode | None = None
    children: dict[int, _MockTreeNode] = field(default_factory=dict)


@dataclass
class _MockRadixCache:
    """Mimics SGLang's `RadixCache.insert / evict / reset` enough for the
    adapter to wire in. Each `insert(token_ids)` adds one TreeNode whose
    `key` is the new tokens and whose parent is whatever path matched."""

    root: _MockTreeNode = field(default_factory=lambda: _MockTreeNode(key=[]))

    def insert(self, token_ids):
        # Naive: append a new leaf under root with the whole token sequence.
        # Real SGLang does prefix matching + branching; for our probe test,
        # what matters is that the wrapped insert fires the probe callback.
        node = _MockTreeNode(key=list(token_ids), parent=self.root)
        # Cache the node so evict can return it.
        self._last_inserted = node
        return node

    def evict(self, n: int = 1):
        node = getattr(self, "_last_inserted", None)
        self._last_inserted = None
        return [node] if node is not None else []

    def reset(self):
        self.root = _MockTreeNode(key=[])


def test_sglang_adapter_attaches_and_fires_on_insert() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    cache = _MockRadixCache()

    originals = attach_to_radix_cache(probe, cache)
    try:
        cache.insert(list(range(8)))
        assert probe.stored_emitted == 2
    finally:
        detach_from_radix_cache(originals, cache)


def test_sglang_adapter_fires_on_evict() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    cache = _MockRadixCache()

    originals = attach_to_radix_cache(probe, cache)
    try:
        cache.insert(list(range(8)))
        cache.evict(1)
        # After insert: 2 STORED. After evict: 1 REMOVED (last block).
        assert probe.stored_emitted == 2
        assert probe.removed_emitted == 1
    finally:
        detach_from_radix_cache(originals, cache)


def test_sglang_adapter_fires_on_reset() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    cache = _MockRadixCache()

    originals = attach_to_radix_cache(probe, cache)
    try:
        cache.insert(list(range(8)))
        cache.reset()
        assert probe.cleared_emitted == 1
    finally:
        detach_from_radix_cache(originals, cache)


def test_sglang_adapter_detach_restores_originals() -> None:
    pub = _StubPublisher()
    probe = _make_probe(publisher=pub, index_block_size=4)
    cache = _MockRadixCache()

    # Behavior-based check (bound-method identity isn't stable across
    # attribute access on the same instance, so compare effects, not refs).
    originals = attach_to_radix_cache(probe, cache)
    cache.insert(list(range(4)))
    assert probe.stored_emitted == 1  # patched, probe fired

    detach_from_radix_cache(originals, cache)
    cache.insert(list(range(4, 8)))
    # Probe doesn't fire after detach — count stays at 1.
    assert probe.stored_emitted == 1


def test_walk_full_prefix_tokens() -> None:
    """The helper walks parent links to assemble the root-to-leaf token sequence."""
    root = _MockTreeNode(key=[])  # treated as root sentinel (key falsy)
    a = _MockTreeNode(key=[1, 2, 3], parent=root)
    b = _MockTreeNode(key=[4, 5], parent=a)
    c = _MockTreeNode(key=[6], parent=b)
    assert _walk_full_prefix_tokens(c) == [1, 2, 3, 4, 5, 6]
    assert _walk_full_prefix_tokens(a) == [1, 2, 3]


def test_walk_full_prefix_tokens_root_only() -> None:
    """Walking from the root sentinel (empty key) returns empty."""
    root = _MockTreeNode(key=[])
    assert _walk_full_prefix_tokens(root) == []
