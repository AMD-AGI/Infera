###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Routing targets: a (worker, dp_rank) pair the router can dispatch to.

Data parallelism exposes ranks in two shapes, distinguished by existing
``WorkerInfo`` fields rather than a dedicated flag:

- **rank-as-endpoint** — each DP rank is its own registered worker (vLLM
  external-LB sets ``dp_rank`` per process; independent replicas leave
  ``dp_size`` unset). One target per worker, ``dp_rank=None``.
- **rank-multiplexed** — one endpoint fronts ``dp_size`` internal ranks
  (SGLang native ``--dp-size``: ``dp_size>1`` with ``dp_rank=None``).
  Expanded into one target per rank, selected at dispatch via a hint.
"""

from __future__ import annotations

from dataclasses import dataclass

from infera.common.worker_pool import WorkerInfo


def is_rank_multiplexed(w: WorkerInfo) -> bool:
    return (w.dp_size or 1) > 1 and w.dp_rank is None


@dataclass(frozen=True)
class RouteTarget:
    worker: WorkerInfo
    dp_rank: int | None = None

    @property
    def route_key(self) -> str:
        """Stable key for per-target bookkeeping (active blocks, metrics).
        Identical to ``worker_id`` for single-rank targets, so endpoint
        workers behave exactly as before."""
        if self.dp_rank is None:
            return self.worker.worker_id
        return f"{self.worker.worker_id}#dp{self.dp_rank}"


def expand_targets(workers: list[WorkerInfo]) -> list[RouteTarget]:
    """One target per worker, except rank-multiplexed workers fan out to
    one target per DP rank."""
    targets: list[RouteTarget] = []
    for w in workers:
        if is_rank_multiplexed(w):
            targets.extend(RouteTarget(w, r) for r in range(w.dp_size))
        else:
            targets.append(RouteTarget(w))
    return targets
