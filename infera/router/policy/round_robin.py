###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging

from infera.common.worker_pool import WorkerInfo
from infera.router.policy.base import Policy
from infera.router.policy.target import RouteTarget, expand_targets

logger = logging.getLogger(__name__)


class RoundRobinPolicy(Policy):
    """Pure round-robin over (worker, dp_rank) targets. Demo-quality; not
    state-sharded across replicas.

    Rank-multiplexed workers fan out via ``expand_targets`` so RR rotates
    over (and pins) each DP rank itself rather than delegating rank choice
    to the engine's internal load balancer — same "expand → pick → pin"
    path as the cost-aware policies, differing only in how they pick.

    Each distinct target list (keyed by the tuple of route keys) keeps its
    own counter. This matters for disaggregated dispatch where one request
    calls ``pick()`` twice -- once for the prefill pool and once for the
    decode pool. A single shared counter would advance by two per request
    and the parity would pin every prefill pick to the same target when
    there are exactly two prefills. The per-pool counter keeps each pool's
    rotation independent.
    """

    def __init__(self) -> None:
        self._counters: dict[tuple[str, ...], int] = {}

    def pick(
        self,
        candidates: list[WorkerInfo],
        request: dict,
        *,
        role_hint: str | None = None,
    ) -> tuple[RouteTarget, list[int]]:
        # role_hint doesn't change round-robin's choice; it's logged so a PD run
        # can bucket picks by pool the same way the cost-aware policy is read.
        targets = expand_targets(candidates)
        key = tuple(t.route_key for t in targets)
        idx = self._counters.get(key, 0)
        self._counters[key] = idx + 1
        target = targets[idx % len(targets)]
        logger.info(
            "pick policy=round-robin role=%s picked=%s", role_hint or "mixed", target.route_key
        )
        # Empty list: stateless w.r.t. KV blocks, signals "no per-request
        # tracking" to the cost-aware lifecycle hooks.
        return target, []
