###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The config sweep is a search's pre-prune step, so it has to behave like one.

Ranking and feasibility are the contract: a caller uses it to decide which few
configs are worth real GPU time, and a sweep that silently drops points or stops
on a bad one would quietly narrow the search instead of ranking it.
"""

from __future__ import annotations

from infera.projection.core.projection.inference_projection.sweep import (
    SweepPoint,
    SweepResult,
    sweep,
    to_json,
)


def _pt(**kw):
    base = dict(tp=1, ep=1, pp=1, concurrency=1, isl=1024, osl=1024)
    base.update(kw)
    return SweepPoint(**base)


def test_ranking_puts_the_best_config_first():
    res = SweepResult(points=[
        _pt(tp=1, decode_tps_per_gpu=100.0),
        _pt(tp=8, decode_tps_per_gpu=900.0),
        _pt(tp=4, decode_tps_per_gpu=400.0),
    ])
    assert [p.tp for p in res.ranked()] == [8, 4, 1]
    assert res.shortlist(2)[0].tp == 8
    # Latency is a minimise objective, so the flag has to work both ways.
    res2 = SweepResult(points=[_pt(tpot_ms=30.0), _pt(tpot_ms=10.0)])
    assert res2.ranked("tpot_ms", maximize=False)[0].tpot_ms == 10.0


def test_infeasible_points_are_kept_but_never_recommended():
    """A search needs "does not fit" and "was not tried" to be different answers."""
    res = SweepResult(points=[
        _pt(tp=1, decode_tps_per_gpu=999.0, feasible=False, reason="needs 900 GB/GPU"),
        _pt(tp=8, decode_tps_per_gpu=100.0),
    ])
    assert len(res.points) == 2, "infeasible points must survive for auditing"
    assert [p.tp for p in res.feasible] == [8]
    assert [p.tp for p in res.shortlist(5)] == [8], (
        "an unrunnable config must not be recommended however fast it projects"
    )


def test_a_failing_config_does_not_abort_the_sweep():
    """One bad point must cost one point, not the whole sweep."""
    res = sweep(
        "gpt_oss_120B",
        tp=(1,), ep=(1,), pp=(1,), concurrency=(8,),
        workload="/nonexistent/workload.yaml",
    )
    assert res.n_projected == 1
    assert res.points[0].feasible is False
    assert res.points[0].reason, "a failed point must say why"


def test_valid_filter_skips_combinations_before_projecting_them():
    calls = []

    def valid(tp, ep, pp):
        calls.append((tp, ep, pp))
        return False

    res = sweep(
        "gpt_oss_120B",
        tp=(1, 2), ep=(1, 2), pp=(1,), concurrency=(8,),
        valid=valid,
    )
    assert res.n_projected == 0, "filtered combinations must not be projected"
    assert len(calls) == 4


def test_serialised_sweep_carries_what_it_cost():
    """The speed claim is the point of the tool, so it travels with the data."""
    res = SweepResult(points=[_pt()], elapsed_s=2.0, n_projected=50)
    blob = to_json(res)
    assert blob["n_projected"] == 50
    assert blob["per_config_ms"] == 40.0
    assert len(blob["points"]) == 1
