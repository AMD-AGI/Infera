###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""msgspec schemas for the SGLang and vLLM KV cache event wire formats.

The two engines serialize kv-cache events DIFFERENTLY, so the router must decode
each worker's stream with the schema matching that worker's engine (see
``batch_type_for_engine`` / ``KvEventClient``). Decoding with the wrong event
encoding raises "Expected array/object, got ..." on every event and the router's
cache view stays empty -> 0% cache overlap for that engine.

* **SGLang** (``python/sglang/srt/disaggregation/kv_events.py``): the KVCacheEvent
  base is ``array_like=True, tag=True`` -> each event is a TAGGED ARRAY; block
  hashes are ``list[int]``. This is the original/historical wire format.
* **vLLM** (``vllm/distributed/kv_events.py``): the KVCacheEvent base is ``tag=True``
  WITHOUT ``array_like`` -> each event is a TAGGED MAP; block hashes are
  ``ExternalBlockHash = int | bytes``.

Both wrap events in an ``array_like=True`` batch (ts, events, dp_rank). The struct
config (``array_like``, ``tag``, tag value) must match upstream byte-for-byte.
"""

from __future__ import annotations

import msgspec

# ---------------------------------------------------------------------------
# vLLM wire format: events are tagged MAPS; hashes are int | bytes.
# ---------------------------------------------------------------------------


class _VllmKVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    pass


class BlockStored(_VllmKVCacheEvent):
    block_hashes: list[int | bytes]
    parent_block_hash: int | bytes | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None = None
    # vLLM emits one event PER KV-CACHE GROUP, and only the attention groups
    # carry a usable hash-per-block. On a hybrid model (Kimi-K3: 3 KDA/Mamba
    # groups + 1 MLA group) the Mamba groups run prefix caching in "align"
    # mode, where all but one block per step is a null block that is skipped
    # when the hash list is built -- while ``token_ids`` still spans the whole
    # range. Measured on Kimi-K3: the Mamba groups report 3840 tokens against
    # ONE hash at block_size=768, the MLA group 3840 against five.
    #
    # There is no field saying which chunk the surviving hash covers, so a
    # Mamba event cannot be indexed at all. Worse, vLLM's block hash does not
    # mix in the group id, so with equal block sizes a Mamba hash COLLIDES with
    # an attention hash and overwrites its entry in the engine-hash -> router-
    # hash map, breaking the parent chain for every later block. Filtering on
    # these two fields is what keeps the one usable stream intact; see
    # ``client._on_block_stored``.
    #
    # Both are absent on SGLang and on vLLM builds predating them, so they
    # default to None and the filter must fail open. Upstream: vllm#44451.
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None


class BlockRemoved(_VllmKVCacheEvent):
    block_hashes: list[int | bytes]
    medium: str | None = None


class AllBlocksCleared(_VllmKVCacheEvent):
    pass


class KVEventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
    attn_dp_rank: int | None = None


# ---------------------------------------------------------------------------
# SGLang wire format: events are tagged ARRAYS; hashes are int.
# Same tag values ("BlockStored"/...) as vLLM, so keep them explicit here since
# the class names differ.
# ---------------------------------------------------------------------------


class _SglangKVCacheEvent(msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True):
    pass


class SglangBlockStored(_SglangKVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    # With EAGLE/MTP the radix key is a bigram view (``RadixKey.is_bigram``, set
    # from ``is_eagle``), and the engine reports a block's tokens as the
    # overlapping pairs ``(t[i], t[i+1])`` instead of bare ints -- so this field
    # is list[int] on a plain engine and list[tuple[int, int]] under MTP. See
    # ``client._flat_tokens`` for how the pairs map back onto flat tokens.
    token_ids: list[int | tuple[int, int]]
    block_size: int
    lora_id: int | None
    medium: str | None = None


class SglangBlockRemoved(_SglangKVCacheEvent, tag="BlockRemoved"):
    block_hashes: list[int]
    medium: str | None = None


class SglangAllBlocksCleared(_SglangKVCacheEvent, tag="AllBlocksCleared"):
    pass


class SglangKVEventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[SglangBlockStored | SglangBlockRemoved | SglangAllBlocksCleared]
    attn_dp_rank: int | None = None


# ---------------------------------------------------------------------------
# Engine dispatch. Event handlers should isinstance-check against these tuples
# (the two families share field names, so access is uniform once dispatched).
# ---------------------------------------------------------------------------

BLOCK_STORED_TYPES = (BlockStored, SglangBlockStored)
BLOCK_REMOVED_TYPES = (BlockRemoved, SglangBlockRemoved)
ALL_CLEARED_TYPES = (AllBlocksCleared, SglangAllBlocksCleared)


def batch_type_for_engine(engine) -> type:
    """KVEventBatch schema matching a worker's engine wire format.

    vLLM emits tagged-MAP events; SGLang (and anything else, the historical
    default) emits tagged-ARRAY events.
    """
    from infera.common.worker_pool import EngineType

    if engine == EngineType.VLLM:
        return KVEventBatch
    return SglangKVEventBatch
