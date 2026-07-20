###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Carry the router's per-request DP-rank choice to SGLang on the wire.

All three helpers no-op for endpoint-addressed targets (``dp_rank is None``,
e.g. vLLM external-LB) and non-SGLang engines, so callers apply them
unconditionally.
"""

from __future__ import annotations

from infera.common.worker_pool import EngineType
from infera.router.policy.target import RouteTarget


def dp_rank_header(target: RouteTarget) -> dict[str, str] | None:
    """``X-Data-Parallel-Rank`` header pinning the request to a DP rank. Both
    SGLang (``DataParallelController``) and vLLM (``_get_data_parallel_rank``,
    case-insensitive) honour it; no-op when the target carries no rank."""
    if target.dp_rank is None:
        return None
    return {"X-Data-Parallel-Rank": str(target.dp_rank)}


def inject_disagg_prefill_dp_rank(
    body: dict, *, prefill_target: RouteTarget, decode_engine: EngineType
) -> dict:
    """Tell the decode worker which prefill rank holds its KV (a fresh dict;
    ``body`` is left untouched)."""
    if prefill_target.dp_rank is None or decode_engine != EngineType.SGLANG:
        return body
    return {**body, "disagg_prefill_dp_rank": prefill_target.dp_rank}


def align_room_to_prefill_rank(room_id: int, prefill_target: RouteTarget) -> int:
    """Rewrite the low residue so ``room_id % dp_size == dp_rank``; the high bits
    stay random so the room is still unique.

    SGLang's default ``follow_bootstrap_room`` balancer derives the prefill rank
    from ``bootstrap_room % dp_size`` and rejects the leg (``KVTransferError``)
    if it landed elsewhere. vLLM-mooncake reuses the same residue as the carrier
    for the steered rank: the decode protocol recovers it as ``room % dp_size``
    to address the prefill rank's ``engine_id`` (``..._dp{K}``).
    """
    w = prefill_target.worker
    if prefill_target.dp_rank is None or not w.dp_size:
        return room_id
    return room_id - (room_id % w.dp_size) + prefill_target.dp_rank
