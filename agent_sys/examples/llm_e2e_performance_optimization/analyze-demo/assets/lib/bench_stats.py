#!/usr/bin/env python3
"""The 5-group weighted average that mission.md section 3.2.7 asks for.

    "性能测试结果：5次加权平均，每次运行loop 10次以上取平均"

Five groups, each an average over at least ten iterations, combined by a
weighted mean. `verify_workset` produces the numbers and `check_workset_runs`
judges them, and both import this module so that the two cannot disagree about
what the weighted average means.

The weight is the iteration count of the group. A group that ran 20 iterations
carries twice the weight of one that ran 10, which is the only weighting that
makes the combined figure equal the mean over all individual iterations.
"""

from __future__ import annotations

import math

MIN_GROUPS = 5
MIN_ITERS_PER_GROUP = 10


class BenchShapeError(ValueError):
    """The measurement does not have the shape mission 3.2.7 requires."""


def check_shape(groups: list[dict]) -> None:
    """Raise unless there are >= 5 groups of >= 10 iterations each."""
    if len(groups) < MIN_GROUPS:
        raise BenchShapeError(
            f"need at least {MIN_GROUPS} measurement groups, got {len(groups)}"
        )
    for index, group in enumerate(groups):
        iters = int(group.get("iters") or 0)
        if iters < MIN_ITERS_PER_GROUP:
            raise BenchShapeError(
                f"group {index} ran {iters} iterations, "
                f"mission 3.2.7 requires at least {MIN_ITERS_PER_GROUP}"
            )
        if group.get("mean_ms") is None:
            raise BenchShapeError(f"group {index} carries no mean_ms")


def weighted_mean(groups: list[dict]) -> float:
    total_weight = sum(int(g["iters"]) for g in groups)
    if total_weight == 0:
        raise BenchShapeError("total iteration count is zero")
    return sum(float(g["mean_ms"]) * int(g["iters"]) for g in groups) / total_weight


def rsd(groups: list[dict]) -> float:
    """Relative standard deviation of the per-group means.

    This is the run-to-run stability figure. A high value means the machine was
    not quiet, and a baseline measured then is not a baseline forge-loop can
    optimize against. Population standard deviation, because the five groups are
    the whole measurement rather than a sample of a larger one.
    """
    means = [float(g["mean_ms"]) for g in groups]
    average = sum(means) / len(means)
    if average == 0:
        return math.inf
    variance = sum((m - average) ** 2 for m in means) / len(means)
    return math.sqrt(variance) / average


def summarize(groups: list[dict]) -> dict:
    """The block written into `workset_evidence` for one operator."""
    check_shape(groups)
    means = [float(g["mean_ms"]) for g in groups]
    return {
        "groups": len(groups),
        "iters_total": sum(int(g["iters"]) for g in groups),
        "weighted_mean_ms": weighted_mean(groups),
        "min_group_ms": min(means),
        "max_group_ms": max(means),
        "rsd": rsd(groups),
        "per_group_ms": means,
    }
