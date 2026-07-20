###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""In-memory KV index: per-(model, compat_key, publisher) radix tree.

Concurrency model: this module's public methods are **synchronous**.
The caller (typically a single writer task draining a ZMQ-fed queue)
is responsible for serializing mutations and for holding read locks
during `find_matches` walks. We do not embed locks here so that:
  - unit tests don't need asyncio
  - the same primitive serves both the in-process single-writer model
    (Phase 1) and any future native (Rust) implementation across an
    FFI boundary

Tree shape: a synthetic root holds children keyed by the *first* block's
block_hash. Each child holds its own children keyed by the next block's
block_hash. Traversal walks block-by-block; the sequence_hash chain is
implicit in the tree path.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from infera.kv.types import BlockKey, OverlapBlocks, Tier

# Key into the top-level trees dict — fully identifies a tree.
IndexKey = tuple[str, str, str]  # (model_name, compat_key, publisher_id)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass
class _IndexNode:
    """One node in the radix tree (one cached block on one publisher)."""

    block_hash: int  # 0 for the synthetic root
    sequence_hash: int  # 0 for the synthetic root
    parent_sequence_hash: int | None  # None for the synthetic root AND for chain heads
    children: dict[int, _IndexNode] = field(default_factory=dict)  # block_hash → child
    tiers: set[Tier] = field(default_factory=set)
    inserted_ts_ms: int = 0
    last_access_ts_ms: int = 0
    # When set, this node is optimistic (placed by routing decision before the
    # publisher's real event arrives). Real `apply_stored` clears it. Eviction
    # sweep removes nodes whose optimistic_until_ms < now_ms with no
    # confirming event.
    optimistic_until_ms: int | None = None

    def is_optimistic(self) -> bool:
        return self.optimistic_until_ms is not None

    def is_root(self) -> bool:
        return self.parent_sequence_hash is None and self.sequence_hash == 0


@dataclass
class IndexStats:
    nodes: int = 0
    confirmed_nodes: int = 0
    optimistic_nodes: int = 0
    publishers: int = 0
    models: int = 0
    compat_keys: int = 0


class KVIndex:
    """Per-(model, compat_key, publisher_id) prefix trees."""

    def __init__(self, default_ttl_s: float = 600.0) -> None:
        self._default_ttl_s = default_ttl_s
        # The "root" for each tree is a synthetic node with no payload.
        self._trees: dict[IndexKey, _IndexNode] = {}
        # Flat O(1) lookup: (key, sequence_hash) → node. Used for apply_removed.
        self._by_seq: dict[tuple[IndexKey, int], _IndexNode] = {}

    # ------------------------------------------------------------------
    # Mutations (called by the single writer task)
    # ------------------------------------------------------------------

    def apply_stored(
        self,
        *,
        publisher_id: str,
        model: str,
        compat_key: str,
        block: BlockKey,
        tier: Tier,
        now_ms: int | None = None,
    ) -> None:
        """Mark one block as stored on the given publisher's tier.

        Idempotent: re-applying the same (block, tier) is a no-op except
        for refreshing `last_access_ts_ms`. If the node was optimistic,
        the confirming real event clears `optimistic_until_ms`.
        """
        now = now_ms if now_ms is not None else _now_ms()
        key = (model, compat_key, publisher_id)
        root = self._ensure_root(key)
        node = self._upsert_node(root, key, block, now)
        node.tiers.add(tier)
        # Real event confirms an optimistic record (if any) and clears the TTL.
        node.optimistic_until_ms = None
        node.last_access_ts_ms = now

    def apply_removed(
        self,
        *,
        publisher_id: str,
        model: str,
        compat_key: str,
        sequence_hash: int,
        tier: Tier,
    ) -> None:
        """Remove one tier mark from a block. If no tiers remain, the node
        is unlinked from its parent and from `_by_seq`. Children are NOT
        recursively removed — they remain reachable via the surviving
        parents (note: an event protocol that issues `removed` for a
        non-leaf is the engine's decision; we just record the residency).

        Idempotent: removing a tier that's already absent is a no-op.
        """
        key = (model, compat_key, publisher_id)
        node = self._by_seq.get((key, sequence_hash))
        if node is None:
            return
        node.tiers.discard(tier)
        if not node.tiers and not node.children:
            # Fully gone: unlink from parent and lookup.
            self._unlink_leaf(key, node)

    def apply_cleared(
        self,
        *,
        publisher_id: str,
        scope: str = "all",
    ) -> None:
        """Drop entries matching `scope`. Forms:
        - "all"                  → all trees for this publisher
        - "model:<name>"         → trees with model == name
        - "tier:<name>"          → only that tier mark (across all of pub's blocks)
        - "compat_key:<hex>"     → trees with compat_key == hex
        """
        # The first three handle a partition of publisher_id; the compat_key
        # form requires walking trees and matching the middle of the key tuple.
        to_drop: list[IndexKey] = []
        for ik in self._trees:
            _model, _ck, _pub = ik
            if _pub != publisher_id:
                continue
            if scope == "all":
                to_drop.append(ik)
            elif scope.startswith("model:") and _model == scope.removeprefix("model:"):
                to_drop.append(ik)
            elif scope.startswith("compat_key:") and _ck == scope.removeprefix("compat_key:"):
                to_drop.append(ik)

        if scope.startswith("tier:"):
            tier_name = scope.removeprefix("tier:")
            try:
                tier = Tier(tier_name)
            except ValueError:
                return
            for ik in [k for k in self._trees if k[2] == publisher_id]:
                self._strip_tier(ik, tier)
            return

        for ik in to_drop:
            self._drop_tree(ik)

    def drop_publisher(self, publisher_id: str) -> None:
        """Remove all index state for a publisher (worker or pool daemon).
        Called when the publisher's etcd lease expires / a deregister event
        fires.
        """
        for ik in [k for k in self._trees if k[2] == publisher_id]:
            self._drop_tree(ik)

    def drop_tree(self, *, model: str, compat_key: str, publisher_id: str) -> None:
        """Wipe one (model, compat_key, publisher_id) tree. Used by the
        snapshot reconciler to rebuild a publisher's state under the
        write lock before applying the fresh snapshot.
        """
        self._drop_tree((model, compat_key, publisher_id))

    def optimistically_record(
        self,
        *,
        publisher_id: str,
        model: str,
        compat_key: str,
        chain: Iterable[BlockKey],
        tier: Tier,
        ttl_s: float = 2.0,
        now_ms: int | None = None,
    ) -> None:
        """Tentatively record an entire chain as cached on `publisher_id`
        before the real event arrives. If a confirming `apply_stored`
        doesn't come within `ttl_s`, eviction sweep removes the entries.

        Used by the PD optimistic-D-promotion path (06-pd-interaction.md).
        TTL default 2 s matches llm-d's speculativeIndexing default — tuned
        to comfortably exceed typical routing-to-event latency.
        """
        now = now_ms if now_ms is not None else _now_ms()
        key = (model, compat_key, publisher_id)
        root = self._ensure_root(key)
        deadline = now + int(ttl_s * 1000)
        for block in chain:
            # Snapshot confirmed-ness BEFORE upsert mutates anything.
            existing = self._by_seq.get((key, block.sequence_hash))
            was_confirmed = (
                existing is not None
                and existing.optimistic_until_ms is None
                and bool(existing.tiers)
            )
            node = self._upsert_node(root, key, block, now)
            node.tiers.add(tier)
            node.last_access_ts_ms = now
            # Set the optimistic deadline unless the node was already
            # confirmed by a real event before this call.
            if not was_confirmed:
                node.optimistic_until_ms = deadline

    # ------------------------------------------------------------------
    # Query (called by routing-policy reader tasks)
    # ------------------------------------------------------------------

    def find_matches(
        self,
        *,
        model: str,
        compat_key: str,
        chain: list[BlockKey],
        candidates: list[str],
        now_ms: int | None = None,
    ) -> dict[str, OverlapBlocks]:
        """For each candidate publisher_id, walk the publisher's tree top-down
        following the chain's block_hashes; count consecutive prefix matches
        per tier.

        Returns one OverlapBlocks per candidate (zero-filled if no tree).
        """
        now = now_ms if now_ms is not None else _now_ms()
        out: dict[str, OverlapBlocks] = {}
        for pub in candidates:
            out[pub] = self._find_one(model, compat_key, pub, chain, now)
        return out

    def _find_one(
        self,
        model: str,
        compat_key: str,
        publisher_id: str,
        chain: list[BlockKey],
        now_ms: int,
    ) -> OverlapBlocks:
        key = (model, compat_key, publisher_id)
        root = self._trees.get(key)
        if root is None or not chain:
            return OverlapBlocks()

        overlap = OverlapBlocks()
        cursor = root
        for block in chain:
            child = cursor.children.get(block.block_hash)
            if child is None:
                break
            # Sanity-check the cumulative sequence_hash to detect hash
            # collisions (rare but documented in 09-open-questions Q5).
            if child.sequence_hash != block.sequence_hash:
                break
            # Count this block per resident tier. A block on multiple tiers
            # counts once per tier (the policy picks the best later).
            if Tier.DEVICE in child.tiers:
                overlap.device += 1
            if Tier.HOST in child.tiers:
                overlap.host += 1
            if Tier.DISK in child.tiers:
                overlap.disk += 1
            if Tier.FABRIC in child.tiers:
                overlap.fabric += 1
            child.last_access_ts_ms = now_ms
            cursor = child
        return overlap

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def evict_expired(self, now_ms: int | None = None) -> int:
        """Sweep TTL-expired nodes. Returns the number of nodes removed.

        Two eviction triggers:
          1. Optimistic nodes whose `optimistic_until_ms` passed without
             being confirmed.
          2. Confirmed nodes whose `last_access_ts_ms` exceeds
             `default_ttl_s` — i.e. the entry hasn't been touched
             recently. PR #9 review fix P1: previously the check was
             against `inserted_ts_ms`, which evicts popular blocks at
             insert-age regardless of how often they're hit. With
             access-age, popular blocks get refreshed on each hit (see
             `find_matches` at line 287) and stay cached.
        """
        now = now_ms if now_ms is not None else _now_ms()
        ttl_ms = int(self._default_ttl_s * 1000)

        # Phase 1: identify all nodes whose own TTL has expired and clear
        # their tier marks. We don't unlink yet — a parent whose children
        # are also expiring must wait for the children to be unlinked first.
        for (_ik, _seq), node in self._by_seq.items():
            if node.optimistic_until_ms is not None and node.optimistic_until_ms < now:
                node.tiers.clear()
            elif node.optimistic_until_ms is None and now - node.last_access_ts_ms > ttl_ms:
                node.tiers.clear()

        # Phase 2: iteratively unlink any node with empty tiers and no
        # children, until no more can be removed. This walks bottom-up
        # naturally: leaves go first, then their parents become leaves, etc.
        removed = 0
        changed = True
        while changed:
            changed = False
            for ik, _seq in list(self._by_seq.keys()):
                node = self._by_seq.get((ik, _seq))
                if node is None:
                    continue
                if not node.tiers and not node.children:
                    self._unlink_leaf(ik, node)
                    removed += 1
                    changed = True
        return removed

    def stats(self) -> IndexStats:
        models: set[str] = set()
        compat_keys: set[str] = set()
        publishers: set[str] = set()
        confirmed = 0
        optimistic = 0
        for model, ck, pub in self._trees:
            models.add(model)
            compat_keys.add(ck)
            publishers.add(pub)
        for node in self._by_seq.values():
            if node.is_optimistic():
                optimistic += 1
            else:
                confirmed += 1
        return IndexStats(
            nodes=len(self._by_seq),
            confirmed_nodes=confirmed,
            optimistic_nodes=optimistic,
            publishers=len(publishers),
            models=len(models),
            compat_keys=len(compat_keys),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_root(self, key: IndexKey) -> _IndexNode:
        if key not in self._trees:
            self._trees[key] = _IndexNode(
                block_hash=0,
                sequence_hash=0,
                parent_sequence_hash=None,
            )
        return self._trees[key]

    def _upsert_node(
        self,
        root: _IndexNode,
        key: IndexKey,
        block: BlockKey,
        now_ms: int,
    ) -> _IndexNode:
        """Walk from root to where this block belongs, creating intermediate
        nodes if missing. Returns the node for `block`.

        If `block.parent_sequence_hash is None`: child of root.
        Otherwise: child of the node identified by parent_sequence_hash
        (looked up via _by_seq for O(1) parent-finding).
        """
        if block.parent_sequence_hash is None:
            parent = root
        else:
            parent = self._by_seq.get((key, block.parent_sequence_hash))
            if parent is None:
                # We received an event whose parent hasn't been seen yet
                # (out-of-order or dropped earlier event). Fall back to
                # rooting it directly — better than dropping. Snapshot
                # reconciliation will repair the structure.
                parent = root

        existing = parent.children.get(block.block_hash)
        if existing is not None:
            existing.last_access_ts_ms = now_ms
            return existing

        node = _IndexNode(
            block_hash=block.block_hash,
            sequence_hash=block.sequence_hash,
            parent_sequence_hash=block.parent_sequence_hash,
            inserted_ts_ms=now_ms,
            last_access_ts_ms=now_ms,
        )
        parent.children[block.block_hash] = node
        self._by_seq[(key, block.sequence_hash)] = node
        return node

    def _unlink_leaf(self, key: IndexKey, node: _IndexNode) -> None:
        """Remove a leaf node from its parent's children and from _by_seq."""
        # Find the parent. Use parent_sequence_hash to look it up.
        if node.parent_sequence_hash is None:
            parent = self._trees.get(key)
        else:
            parent = self._by_seq.get((key, node.parent_sequence_hash))
        if parent is not None:
            parent.children.pop(node.block_hash, None)
        self._by_seq.pop((key, node.sequence_hash), None)

    def _drop_tree(self, key: IndexKey) -> None:
        """Drop an entire tree, including all entries in _by_seq for it."""
        if key not in self._trees:
            return
        # Clear lookup entries with matching (model, compat_key, publisher_id).
        for ik, seq in [k for k in self._by_seq if k[0] == key]:
            self._by_seq.pop((ik, seq), None)
        self._trees.pop(key, None)

    def _strip_tier(self, key: IndexKey, tier: Tier) -> None:
        """Remove `tier` from every node in a tree. Nodes that lose all tiers
        and have no children are unlinked.
        """
        # Iterate nodes via _by_seq for this key.
        to_unlink: list[_IndexNode] = []
        for (ik, _seq), node in self._by_seq.items():
            if ik != key:
                continue
            node.tiers.discard(tier)
            if not node.tiers and not node.children:
                to_unlink.append(node)
        for node in to_unlink:
            self._unlink_leaf(key, node)

    def _is_already_confirmed(self, node: _IndexNode) -> bool:
        return node.optimistic_until_ms is None and node.tiers
