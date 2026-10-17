###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Carry the router's per-request DP-rank choice to the engine on the wire.

SGLang and vLLM read the rank from a header, ATOM from the request body
(``data_parallel_rank``). Every helper no-ops for endpoint-addressed targets
(``dp_rank is None``, e.g. vLLM external-LB) and for engines it does not
apply to, so callers apply them unconditionally.
"""

from __future__ import annotations

from infera.common.worker_pool import EngineType
from infera.router.policy.target import RouteTarget

#: Both engines honour this (SGLang ``DataParallelController``, vLLM
#: ``_get_data_parallel_rank``, case-insensitively). Named rather than inlined
#: so anything asserting on it breaks loudly on a rename instead of silently
#: never matching. Mirrors Rust's DP_RANK_HEADER.
DP_RANK_HEADER = "X-Data-Parallel-Rank"

#: ATOM's request field for the same thing (``ChatCompletionRequest`` /
#: ``CompletionRequest.data_parallel_rank``); it does not read the header.
#: An explicit rank wins over ATOM's own load balancing and session affinity.
DP_RANK_BODY_FIELD = "data_parallel_rank"


def _rank_in_body(target: RouteTarget) -> bool:
    return target.worker.engine == EngineType.ATOM


def dp_rank_header(target: RouteTarget) -> dict[str, str] | None:
    """``X-Data-Parallel-Rank`` header pinning the request to a DP rank. Both
    SGLang (``DataParallelController``) and vLLM (``_get_data_parallel_rank``,
    case-insensitive) honour it; no-op when the target carries no rank or its
    engine reads the rank from the body (see :func:`inject_dp_rank`)."""
    if target.dp_rank is None or _rank_in_body(target):
        return None
    return {DP_RANK_HEADER: str(target.dp_rank)}


def inject_dp_rank(body: dict, target: RouteTarget) -> dict:
    """Pin an ATOM target's DP rank in the body (a fresh dict; ``body`` is
    left untouched). No-op for other engines and rank-less targets."""
    if target.dp_rank is None or not _rank_in_body(target):
        return body
    return {**body, DP_RANK_BODY_FIELD: target.dp_rank}


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
