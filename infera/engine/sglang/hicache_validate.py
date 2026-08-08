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


# Threshold below which a host pool stops being a useful L2.
#
# On sglang <= 0.5.14 it is worse than useless — cache_controller computes
#     prefetch_capacity_limit = max(0, int(0.8 * (host_pool - device_pool)))
# which in token units is 0.8 * device_pool_tokens * (hicache_ratio - 1), so at
# ratio 1.0 the limit is 0, every prefetch is rate-limited, and L3 storage is
# never queried. Empirically on MiniMax-M2.5 TP=1 at ratio 1.0: 0 EXISTS and
# 0 GET to kvd across a 5-minute bench, versus 35 EXISTS + 2048 GET at 2.0.
#
# 0.5.15 changed it to `int(0.5 * host_pool)` (still so on main), which is
# positive for any host pool, so there the low-ratio failure is a hit-rate loss —
# an L2 that evicts about as fast as the L1 it shadows — rather than prefetch being
# switched off. Worth guarding on both; only the older base loses reads entirely.
#
# 1.5 rather than 1.0: the limit is 0 only exactly at 1.0 on the old formula, but a
# host pool under 1.5× the device pool is too thin a margin to be worth an L3 round
# trip on either formula, and it is never what an operator meant to configure.
HICACHE_RATIO_DANGER_THRESHOLD = 1.5


def warn_if_hicache_prefetch_disabled(server_args: Any) -> bool:
    """Log a CRITICAL warning when server_args leaves L3 prefetch useless.

    Returns True iff a warning was emitted (for testability).

    A ratio below the threshold above sizes the host pool close to the device
    pool, which at exactly 1.0 caps `prefetch_capacity_limit` at 0 on older
    SGLang and otherwise leaves an L2 too small to be worth reading. This is the
    infera-side guard so operators don't ship a config that looks like "kvd is
    wired up" but never reads from it.
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
        "cripples SGLang's L3 prefetch. On sglang <= 0.5.14 it DISABLES it: "
        "cache_controller's prefetch_capacity_limit = "
        "0.8 * (host_pool - device_pool) evaluates to ~0 as the ratio "
        "approaches 1.0, so prefetch_rate_limited() returns True on every "
        "attempt and the kvd / mooncake / nixl backend takes writes but is "
        "never read, even when the GPU cache overflows. From 0.5.15 the limit "
        "is 0.5 * host_pool, so prefetch still runs but against a host pool "
        "that evicts as fast as the GPU pool it shadows. "
        "Recommended: --hicache-ratio >= 2.0 (the SGLang default), "
        "or --hicache-size=<GB> to size the host pool outright. "
        "See infera PD design §5.4 for the MI355X TP=1 measurement.",
        ratio,
        storage_backend,
    )
    return True
