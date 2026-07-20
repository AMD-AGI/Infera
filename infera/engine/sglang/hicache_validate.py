###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Hicache-config validation helpers.

Kept in its own module (no sglang import at module load) so unit
tests can exercise the logic without sglang installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Threshold below which SGLang's prefetch becomes effectively disabled.
# Derivation: cache_controller.py:462 computes
#     prefetch_capacity_limit = max(0, int(0.8 * (host_pool - device_pool)))
# In token units this evaluates to
#     0.8 * device_pool_tokens * (hicache_ratio - 1)
# So when hicache_ratio == 1.0 the limit is 0 — every prefetch attempt is
# rate-limited → kvd / mooncake / nixl L3 storage never gets queried.
# Empirically observed on MiniMax-M2.5 TP=1 with --hicache-ratio 1.0:
# 0 EXISTS, 0 GET to kvd across a 5-minute bench, while the same workload
# at ratio=2.0 produced 35 EXISTS + 2048 GET calls.
HICACHE_RATIO_DANGER_THRESHOLD = 1.5


def warn_if_hicache_prefetch_disabled(server_args: Any) -> bool:
    """Log a CRITICAL warning if the server_args config will silently
    disable SGLang's L3 prefetch path.

    Returns True iff a warning was emitted (for testability).

    See `--hicache-ratio` documentation in SGLang's server_args; the
    `prefetch_capacity_limit` formula scales with `(host - device)`,
    so values near 1.0 produce a 0-limit cap and silently disable
    prefetch. Upstream fix tracked separately; this is the
    infera-side guard so operators don't ship configs that look
    like "kvd is wired up" but never actually fetch from it.
    """
    enable_hicache = bool(getattr(server_args, "enable_hierarchical_cache", False))
    storage_backend = getattr(server_args, "hicache_storage_backend", None)
    if not enable_hicache or not storage_backend:
        return False  # not using hicache storage path; warning irrelevant

    # `hicache_size` (in GB) overrides ratio when set. We can't know the
    # GPU pool size at arg-parse time, so for the override case we bail
    # rather than warn falsely.
    size_gb = float(getattr(server_args, "hicache_size", 0) or 0)
    if size_gb > 0:
        return False

    ratio = float(getattr(server_args, "hicache_ratio", 2.0))
    if ratio >= HICACHE_RATIO_DANGER_THRESHOLD:
        return False

    logger.critical(
        "infera: --hicache-ratio=%g with --hicache-storage-backend=%s "
        "will silently DISABLE SGLang L3 prefetch. "
        "Inside SGLang's cache_controller.py the formula "
        "prefetch_capacity_limit = 0.8 * (host_pool - device_pool) "
        "evaluates to ~0 when ratio is close to 1.0, which makes "
        "prefetch_rate_limited() return True on every attempt. "
        "Net effect: kvd / mooncake / nixl backend receives writes "
        "but never any reads, even when GPU cache overflows. "
        "Recommended: --hicache-ratio >= 2.0 (the SGLang default), "
        "or use --hicache-size=<GB> to override the ratio-based "
        "computation. See infera PD design §5.4 for empirical "
        "evidence on MI355X TP=1.",
        ratio,
        storage_backend,
    )
    return True
