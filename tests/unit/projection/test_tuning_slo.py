###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Latency budgets must gate the serving search the way the memory cap does.

Serving is a constrained problem -- maximize throughput subject to a latency
promise -- but the agent optimized a single scalar, so ``max_throughput`` always
walked to the largest batch that fit in HBM and the interactive configuration
never showed up in the results at all.
"""

from __future__ import annotations

from infera.projection.agents.tuning_agent.evaluator import EvalResult, _slo_violations


def _result(**kw) -> EvalResult:
    return EvalResult(legal=True, **kw)


def test_no_slo_configured_rejects_nothing():
    assert _slo_violations(_result(ttft_ms=9999.0, itl_ms=9999.0), {}) == []


def test_null_budget_is_unconstrained():
    """A key present but unset must not read as a budget of zero."""
    slo = {"ttft_ms": None, "tpot_ms": "", "request_latency_ms": None}
    assert _slo_violations(_result(ttft_ms=800.0, itl_ms=40.0), slo) == []


def test_config_inside_budget_passes():
    slo = {"ttft_ms": 500, "tpot_ms": 25}
    assert _slo_violations(_result(ttft_ms=310.0, itl_ms=18.0), slo) == []


def test_ttft_overshoot_is_reported_with_both_numbers():
    misses = _slo_violations(_result(ttft_ms=820.5, itl_ms=18.0), {"ttft_ms": 500})
    assert len(misses) == 1
    assert "820.5" in misses[0] and "500.0" in misses[0] and "TTFT" in misses[0]


def test_tpot_reads_the_projected_itl():
    misses = _slo_violations(_result(ttft_ms=100.0, itl_ms=40.0), {"tpot_ms": 25})
    assert misses and "TPOT" in misses[0]


def test_every_missed_budget_is_listed():
    slo = {"ttft_ms": 500, "tpot_ms": 25, "request_latency_ms": 5000}
    misses = _slo_violations(
        _result(ttft_ms=820.0, itl_ms=40.0, request_latency_ms=12000.0), slo
    )
    assert len(misses) == 3


def test_missing_metric_cannot_fail_a_budget():
    """A projection that never reported the metric is not evidence of a miss."""
    assert _slo_violations(_result(ttft_ms=None), {"ttft_ms": 500}) == []


def test_unknown_slo_key_is_ignored():
    assert _slo_violations(_result(ttft_ms=800.0), {"not_a_metric_ms": 1}) == []
