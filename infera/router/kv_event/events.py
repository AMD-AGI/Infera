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
    token_ids: list[int]
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
