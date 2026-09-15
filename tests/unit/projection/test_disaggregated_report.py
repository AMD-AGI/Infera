###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What a disaggregated projection reports about itself.

The split was modelled and then described in a report that omitted the numbers
the search compares topologies on, which is a quieter failure than a wrong
number: an absent metric is not scored, so the objective is decided among the
candidates that did report one and the winner looks unanimous.

Two omissions, both on every disaggregated row. The step decomposition was
missing entirely, because the path never set the continuous-batching extras --
so ``decode_step_ms_pure``, ``mixed_step_fraction_pct`` and
``tpot_pollution_pct`` were unavailable for exactly the topology whose argument
is that it has no mixed steps. And no line named the GPUs a replica costs,
because the report printed one line per pool and no total, leaving the
GPUs-per-replica objective unscorable for a split while every colocated rival
reported it.

These tests read the report through ``parse_inference_metrics`` -- the function
the search itself parses with -- so they fail if the number is computed but
printed in a shape the search cannot read.
"""

from __future__ import annotations

import pytest

from infera.projection.agents.tuning_agent.inference_tuning import (
    parse_inference_metrics,
)

SPLIT = dict(
    disaggregate=True,
    prefill_tp=4,
    tp=4,
    ep=1,
    prefill_replicas=1,
    decode_replicas=1,
    input_len=2048,
    output_len=128,
    concurrency=32,
)


def _split(**overrides):
    from .conftest import project_spec

    return project_spec(**{**SPLIT, **overrides})


def _colocated(**overrides):
    from .conftest import project_spec

    return project_spec(
        **{
            "tp": 8,
            "ep": 1,
            "input_len": 2048,
            "output_len": 128,
            "concurrency": 32,
            **overrides,
        }
    )


def test_a_split_reports_a_step_decomposition_at_all():
    """The omission, stated as the three objectives it removed from the search."""
    m = parse_inference_metrics(_split()["report"])
    for key in ("decode_step_ms_pure", "mixed_step_fraction_pct", "tpot_pollution_pct"):
        assert m.get(key) is not None, (
            f"{key} is absent from a disaggregated report, so the objective is "
            "decided among colocated candidates only"
        )


def test_a_split_reports_no_mixed_steps_because_that_is_the_point():
    """Moving prefill off the decode GPUs is the whole topology.

    A fraction of exactly zero is a result. Reporting nothing is the search
    being unable to see the result.
    """
    m = parse_inference_metrics(_split()["report"])
    assert m["mixed_step_fraction_pct"] == pytest.approx(0.0)
    assert m["tpot_pollution_pct"] == pytest.approx(0.0)


def test_the_colocated_reference_does_pay_for_mixed_steps():
    """The control that makes the zero above worth reporting.

    If the colocated path also came out at zero, the split would be winning an
    objective nobody loses and the comparison would say nothing.
    """
    m = parse_inference_metrics(_colocated()["report"])
    assert m["mixed_step_fraction_pct"] > 0.0, (
        "a colocated engine lands prefill chunks on decode steps; if this is "
        "zero the comparison with the split is vacuous"
    )


def test_the_cost_of_a_step_that_never_happens_is_the_step_not_zero():
    """``decode_step_ms_mixed`` is minimized, so a zero is a free win.

    There is no mixed step in a split to price. Reporting the pure step says
    that a decode step costs what a decode step costs; reporting zero would
    rank the topology first on a metric it simply does not have.
    """
    m = parse_inference_metrics(_split()["report"])
    assert m["decode_step_ms_pure"] > 0.0
    assert m.get("decode_step_ms_mixed") == pytest.approx(m["decode_step_ms_pure"])


def test_a_split_reports_the_gpus_both_pools_cost():
    """One line per pool names the layout but no figure the search can read."""
    m = parse_inference_metrics(_split(prefill_tp=4, tp=4, decode_replicas=1)["report"])
    assert m.get("replica_gpus") == 8, (
        "a prefill pool of 4 beside one decode replica of 4 costs 8 GPUs; "
        f"got {m.get('replica_gpus')!r}"
    )


def test_the_figure_counts_decode_replicas_rather_than_one_pool():
    """Billing the split for its decode pool alone hides the prefill GPUs."""
    m = parse_inference_metrics(_split(prefill_tp=4, tp=2, decode_replicas=2)["report"])
    assert m.get("replica_gpus") == 8, (
        f"prefill 4 + two decode replicas of 2 is 8 GPUs; got {m.get('replica_gpus')!r}"
    )


def test_a_colocated_run_still_reports_its_own_replica_gpus():
    """The regression guard: the split's line must not displace this one."""
    m = parse_inference_metrics(_colocated(tp=8)["report"])
    assert m.get("replica_gpus") == 8
