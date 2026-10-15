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


def test_a_split_prices_admission_against_its_decode_pool():
    """A split could over-subscribe KV and reported no admission wait at all.

    The pool binds on the decode side, so the split is as capable of
    over-subscribing it as a colocated engine. The term was only ever applied
    on the colocated path, which left a split's TTFT as pure service time --
    and silently, because an absent extra reads the same as a zero one.
    """
    e = _split(kv_pool_tokens=4 * 2048, input_len=2048, concurrency=32)["extras"]
    assert e.get("admission_ceiling") == pytest.approx(4, abs=1), (
        f"a pool holding four 2048-token residents admits 4; got {e.get('admission_ceiling')!r}"
    )
    assert (e.get("admission_wait_ms") or 0.0) > 0.0, (
        "32 clients against a pool that admits 4 must wait to be admitted"
    )


def test_a_roomy_decode_pool_leaves_a_split_with_no_admission_wait():
    """Zero is the answer for the published splits, not a missing term.

    Moving prefill off the decode GPUs is what buys the room: the measured
    DeepSeek splits have a decode pool admitting ~770 sequences against the
    256 ever offered, so they never reach a knee. Their TTFT still grows 9x
    across that range, which is the prefill pool's queue and not this term.
    """
    e = _split(kv_pool_tokens=4096 * 2048, input_len=2048, concurrency=32)["extras"]
    assert e.get("admission_ceiling", 0) > 32
    assert (e.get("admission_wait_ms") or 0.0) == pytest.approx(0.0)


def test_admission_is_charged_per_decode_replica():
    """Clients spread over the pool, so one replica faces its share of them."""
    pool = 8 * 2048  # admits seven of these residents: prompt plus generation
    one = _split(kv_pool_tokens=pool, decode_replicas=1, concurrency=32)["extras"]
    four = _split(kv_pool_tokens=pool, decode_replicas=4, concurrency=32)["extras"]
    eight = _split(kv_pool_tokens=pool, decode_replicas=8, concurrency=32)["extras"]
    assert (one.get("admission_wait_ms") or 0.0) > 0.0
    assert four["admission_wait_ms"] < one["admission_wait_ms"] / 3, (
        "32 clients over four replicas is 8 each and not 32, so one replica's "
        "over-subscription falls roughly four-fold"
    )
    assert (eight.get("admission_wait_ms") or 0.0) == pytest.approx(0.0), (
        "four clients each is inside a pool that admits seven"
    )


def test_a_colocated_run_still_prices_its_own_admission():
    """The regression guard: the new per-pool argument defaults to one pool."""
    e = _colocated(kv_pool_tokens=4 * 2048, input_len=2048, concurrency=32)["extras"]
    assert (e.get("admission_wait_ms") or 0.0) > 0.0
