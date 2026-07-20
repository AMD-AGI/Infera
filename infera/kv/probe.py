###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine-agnostic KV event probe.

Sits between an engine-specific adapter (e.g., the SGLang RadixCache hook
in `infera/engine/sglang/kv_probe.py`) and the publisher + snapshot
producer. Does the coalescing from engine-block events to index-block
events in one tight loop using xxhash_rust (releases the GIL).

Coalescing is per-callback, not per-token. The adapter hooks at the
engine's node-level callback (SGLang `RadixCache.insert`, vLLM
`BlockPool.kv_event_queue` drain) so Python dispatch cost amortizes
across the whole node. This is what makes `engine_block_size=1` (ROCm
AITER) efficient — see 14-performance.md § "Probe coalescing strategy".

Idempotent: re-emitting an event with the same `sequence_hash` is
harmless — the index's `apply_stored` and `apply_removed` are idempotent.
The probe additionally dedupes via a local `_emitted` set to reduce
ZMQ traffic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from infera.kv.hashing import hash_token_blocks
from infera.kv.publisher import KvEventPublisher
from infera.kv.snapshot import SnapshotProducer
from infera.kv.types import MmRun, Tier
from infera.kv.wire import (
    make_cleared_event,
    make_removed_event,
    make_stored_event,
)

logger = logging.getLogger(__name__)


class KvEventProbe:
    """Engine-agnostic adapter that coalesces engine-level cache
    events into index-block events.

    Lifecycle:

        probe = KvEventProbe(
            publisher=pub,
            snapshot_producer=producer,
            model_name="Qwen3.6",
            compat_key="abcdef0123456789",
            index_block_size=64,
            tier=Tier.DEVICE,
        )

        # Engine adapter calls this whenever a new node is cached.
        # `full_prefix_tokens` is the FULL token sequence from the root
        # of the engine's trie to (and including) the new node — needed
        # for the sequence_hash chain to be correct.
        probe.on_node_inserted(full_prefix_tokens=[...])

        probe.on_node_evicted(full_prefix_tokens=[...])

        probe.on_clear()

    Concurrency: methods are sync. The adapter typically calls them from
    one task (the engine's callback), but the underlying publisher's
    `emit` is also sync, so calls from multiple tasks are safe.
    """

    def __init__(
        self,
        *,
        publisher: KvEventPublisher,
        snapshot_producer: SnapshotProducer | None,
        model_name: str,
        compat_key: str,
        index_block_size: int,
        tier: Tier = Tier.DEVICE,
        role: str = "indexable",
        group_idx: int = 0,
        pool_id: str | None = None,
        pool_type: str | None = None,
    ) -> None:
        if index_block_size <= 0:
            raise ValueError(f"index_block_size must be positive, got {index_block_size}")
        self._publisher = publisher
        self._snapshot = snapshot_producer
        self._model_name = model_name
        self._compat_key = compat_key
        self._index_block_size = index_block_size
        self._tier = tier
        self._role = role
        self._group_idx = group_idx
        self._pool_id = pool_id
        self._pool_type = pool_type
        # sequence_hash values we've already emitted as stored. Bounded
        # by the size of the engine's cache state (events arrive for
        # every block; eviction removes them from the set).
        self._emitted: set[int] = set()
        # Metrics
        self.stored_emitted = 0
        self.removed_emitted = 0
        self.cleared_emitted = 0
        self.callbacks_received = 0

    # ------------------------------------------------------------------
    # Engine adapter API
    # ------------------------------------------------------------------

    def on_node_inserted(
        self,
        *,
        full_prefix_tokens: list[int],
        mm_runs: Iterable[MmRun] = (),
        lora_name: str | None = None,
    ) -> None:
        """The engine cached a new node. `full_prefix_tokens` is the
        full chain of tokens from the engine's trie root through this
        node — required to compute correct sequence_hash chains.

        Re-hashing the full prefix on every call is by design: it
        guarantees correctness regardless of how the adapter reports
        partial state. For typical workloads (one `insert` per prefill),
        this is one xxhash pass per request — sub-millisecond even on
        long prompts. See 14-performance.md.
        """
        self.callbacks_received += 1
        chain = hash_token_blocks(
            full_prefix_tokens,
            self._index_block_size,
            mm_runs=tuple(mm_runs),
            lora_name=lora_name,
        )
        for block in chain:
            if block.sequence_hash in self._emitted:
                continue
            event = make_stored_event(
                sequence_hash=block.sequence_hash,
                block_hash=block.block_hash,
                parent_sequence_hash=block.parent_sequence_hash,
                tier=self._tier.value,
                role=self._role,
                group_idx=self._group_idx,
                pool_id=self._pool_id,
                pool_type=self._pool_type,
            )
            self._publisher.emit(
                model=self._model_name,
                compat_key=self._compat_key,
                event=event,
            )
            if self._snapshot is not None:
                self._snapshot.on_event(
                    model=self._model_name,
                    compat_key=self._compat_key,
                    event=event,
                )
            self._emitted.add(block.sequence_hash)
            self.stored_emitted += 1

    def on_node_evicted(
        self,
        *,
        full_prefix_tokens: list[int],
        mm_runs: Iterable[MmRun] = (),
        lora_name: str | None = None,
    ) -> None:
        """The engine evicted a node. `full_prefix_tokens` is the full
        chain from root through the evicted node.

        Conservative approach: emit REMOVED only for the **last** block
        in the chain — the one uniquely owned by this node (blocks
        earlier in the chain may be shared with sibling subtrees that
        are still cached). Under-emission means stale index entries get
        TTL'd or snapshot-reconciled. Over-emission would incorrectly
        invalidate live siblings.

        Engine adapters that have more precise eviction information
        (e.g., a known range that's been entirely freed) can call this
        with just that range's tokens by computing the chain from the
        appropriate prefix.
        """
        self.callbacks_received += 1
        chain = hash_token_blocks(
            full_prefix_tokens,
            self._index_block_size,
            mm_runs=tuple(mm_runs),
            lora_name=lora_name,
        )
        if not chain:
            return
        last = chain[-1]
        if last.sequence_hash not in self._emitted:
            return
        event = make_removed_event(
            sequence_hash=last.sequence_hash,
            tier=self._tier.value,
            role=self._role,
            group_idx=self._group_idx,
            pool_id=self._pool_id,
            pool_type=self._pool_type,
        )
        self._publisher.emit(
            model=self._model_name,
            compat_key=self._compat_key,
            event=event,
        )
        if self._snapshot is not None:
            self._snapshot.on_event(
                model=self._model_name,
                compat_key=self._compat_key,
                event=event,
            )
        self._emitted.discard(last.sequence_hash)
        self.removed_emitted += 1

    def on_clear(self, *, scope: str | None = None) -> None:
        """Emit a `cleared` event. Default scope is this probe's compat_key.

        Engine cache controllers call this when they wipe their cache
        (e.g., on a hot config reload).
        """
        self.callbacks_received += 1
        emit_scope = scope or f"compat_key:{self._compat_key}"
        event = make_cleared_event(scope=emit_scope)
        self._publisher.emit(
            model=self._model_name,
            compat_key=self._compat_key,
            event=event,
        )
        if self._snapshot is not None:
            self._snapshot.on_event(
                model=self._model_name,
                compat_key=self._compat_key,
                event=event,
            )
        # Drop our dedup set — anything we've emitted may have been
        # cleared at the engine.
        if emit_scope == "all" or emit_scope == f"compat_key:{self._compat_key}":
            self._emitted.clear()
        self.cleared_emitted += 1
