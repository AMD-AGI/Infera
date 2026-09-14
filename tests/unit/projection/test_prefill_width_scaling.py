###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""How far prefill shrinks when an anchor is carried to a wider target.

An anchor measured at one width cannot say how the model's per-token prefill
work scales, so carrying it to another width has to either measure the split
from a second anchor or assume one. Ideal sharding is the assumption, and it is
right often enough to be tempting: DeepSeek-V4-Pro, harvested at TP4 and TP8,
shards its per-token prefill by 0.4975 against the 0.5 that ideal predicts.

GLM-5.2-MXFP4 shards by 0.82. Its per-token work is MXFP4 expert GEMMs already
narrow at TP4, and halving them again buys far less than half the time. Every
one of its 56 corpus configs is TP8 scored against a TP4 anchor, so the
assumption applied to all of them at once and read TTFT 39% low, TPOT 19% low
and throughput 27% high, with 1 of 56 inside 10%.
"""

from __future__ import annotations

from types import SimpleNamespace

from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
)


def _ratio(
    bench_tp,
    tgt_tp,
    fit=None,
    restore=True,
    origami=None,
    mode="fit",
    probed=None,
    curves=None,
    budget=0,
    conc=0,
):
    """``_prefill_shard_ratio`` with only the four fields it reads populated.

    Built without ``__init__`` deliberately: the method is a function of the
    anchor widths and the scaling fit, and constructing a whole projector to
    exercise it would test the constructor instead.
    """
    p = InferencePerformanceProjector.__new__(InferencePerformanceProjector)
    p._restore = restore
    p._bench_tp = bench_tp
    p._tgt_tp = tgt_tp
    p._bench_scaling_fit = {"prefill": fit} if fit else {}
    p._measured_width_ratio = InferencePerformanceProjector._measured_width_ratio.__get__(p)
    p._scaling_mode = mode
    p._origami_steps = origami if callable(origami) else lambda b, n, ph: origami
    p._bench_prefill_curves = curves or {}
    p._step_tokens = InferencePerformanceProjector._step_tokens.__get__(p)
    p._measured_ratio_at = InferencePerformanceProjector._measured_ratio_at.__get__(p)
    p._meas_ref_input = 8192
    p.cfg = SimpleNamespace(
        request_config=SimpleNamespace(
            input_seq_len=8192, max_num_batched_tokens=budget, max_concurrency=conc, batch_size=0
        )
    )
    return p._prefill_width_scale(probed)


def test_a_width_matched_anchor_is_not_scaled_at_all():
    """No restore, no scaling -- the anchor already describes the target."""
    assert _ratio(4, 4, restore=False) == (1.0, 1.0)


def test_one_anchor_width_falls_back_to_ideal_sharding(capsys):
    """And says so, because the number is an assumption rather than a reading."""
    assert _ratio(4, 8) == (0.5, 1.0)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "assumption, not a measurement" in out


def test_two_anchor_widths_are_believed_over_the_ideal(capsys):
    """A model whose prefill barely shards is read as barely sharding.

    The fit is ``shardable / tp + invariant``. With 100 ms of shardable work
    against a 400 ms invariant floor, going TP4 -> TP8 moves 425 -> 412.5 ms,
    a ratio of 0.970 rather than the 0.5 ideal would claim.
    """
    got, fixed = _ratio(4, 8, fit={16: (100.0, 400.0)})
    assert abs(got - (412.5 / 425.0)) < 1e-9
    assert got > 0.9
    assert fixed == 1.0
    assert "fitted through two anchor widths" in capsys.readouterr().out


def test_a_model_that_really_does_shard_still_reads_as_sharding():
    """The measured path is not biased towards poor scaling.

    All shardable and no floor reproduces ideal exactly, which is what keeps
    this change a no-op for DeepSeek-V4-Pro rather than a penalty applied to
    every model to fix one.
    """
    assert abs(_ratio(4, 8, fit={16: (100.0, 0.0)})[0] - 0.5) < 1e-9


def test_batch_points_are_averaged_rather_than_one_being_picked():
    """The rate this multiplies is itself averaged over the sweep's batches, so
    the ratio has to describe the same set of points."""
    both, _f = _ratio(4, 8, fit={1: (100.0, 0.0), 16: (100.0, 400.0)})
    assert abs(both - (0.5 + 412.5 / 425.0) / 2) < 1e-9


def test_a_degenerate_fit_does_not_produce_a_zero_or_negative_ratio():
    """A fit with no positive time at either width is skipped, not divided by."""
    assert _ratio(4, 8, fit={16: (0.0, 0.0)}) == (0.5, 1.0)


def test_the_simulator_ratio_is_preferred_over_both_and_moves_the_whole_step():
    """Origami wins because it evaluates the target width instead of
    extrapolating towards it.

    GLM-5.2-MXFP4's anchors sit at TP2 and TP4, between which it shards at
    0.563 -- the fit used here -- while above TP4 it falls to 0.825. Fitting
    through the two visible widths reads -25.8% at TP8 and the simulator's
    ratio reads -6.9%, because the breakdown is not present in anything
    measured below the target. The fit sits below the simulator here, which is
    the usual arrangement, so the floor does not apply.

    A simulator that returns the same step at every prompt length cannot be
    split, so this falls back to moving the whole step by one ratio.
    """
    rate, fixed = _ratio(4, 8, fit={16: (100.0, 3.6)}, origami=(76.8, 100.0), mode="origami")
    assert abs(rate - 0.768) < 1e-9
    assert rate == fixed


def test_a_missing_simulator_falls_back_to_the_measured_fit():
    """Origami is unavailable for some architectures; that is a fallback, not a
    failure."""
    rate, fixed = _ratio(4, 8, fit={16: (100.0, 400.0)}, origami=None, mode="origami")
    assert abs(rate - (412.5 / 425.0)) < 1e-9
    assert fixed == 1.0


def test_the_split_is_reported_but_the_whole_step_ratio_is_what_moves(capsys):
    """Both halves still move together, and the split is logged beside them.

    Carrying the halves separately is the obvious improvement and it loses.
    The simulator has DeepSeek-V4-Pro's per-token work at 0.665 against 0.365
    measured, and its host cost at 0.520 against 1.630 -- both backwards, and
    opposed, so only the blended ratio survives. Splitting reads +14.7% there
    while gaining just +4.9% -> +1.6% on Llama, whose intercept is 8% of a step
    and so cannot tell the laws apart.

    Here the simulator costs 60 ms + 20 us/tok at the anchor width and
    30 ms + 16 us/tok at the target: per-token 0.8, intercept 0.5, and the
    whole step at 8192 tokens 161.072 / 223.84.
    """

    def sim(_batch, n, _phase):
        return (30.0 + 0.016 * n, 60.0 + 0.020 * n)

    rate, fixed = _ratio(4, 8, origami=sim, mode="origami", probed=[1024, 2816, 4608, 6400, 8192])
    assert abs(rate - 161.072 / 223.84) < 1e-9
    assert rate == fixed
    out = capsys.readouterr().out
    assert "per-token x0.800 (20.000 -> 16.000 us/tok)" in out
    assert "per-step x0.500 (60.00 -> 30.00 ms)" in out
    assert "for the record only" in out


def test_a_simulator_slope_that_comes_out_flat_does_not_scale_prefill_away():
    """A zero or falling slope is degenerate, and dividing by it would hand back
    a ratio of zero and delete the model's prefill. The whole-step ratio is used
    instead."""

    def flat(_batch, n, _phase):
        return (50.0, 100.0)

    rate, fixed = _ratio(4, 8, origami=flat, mode="origami", probed=[1024, 8192])
    assert abs(rate - 0.5) < 1e-9
    assert rate == fixed


def test_the_simulator_cannot_claim_better_sharding_than_the_anchors_showed(capsys):
    """The measured widths are a floor on the ratio, never a ceiling.

    Spreading a model wider only ever adds collectives and narrows every GEMM
    again, so the ratio rises with width: measured per-token prefill carries at
    0.580 from TP2 to TP4 on Llama-3.1-8B and then 0.625 from TP4 to TP8, and
    GLM-5.2-MXFP4 goes 0.507 and then 0.68 or worse. A simulator ratio below
    what two anchor widths already demonstrated is claiming a gain never
    observed, so it is raised.

    Here the fit reads 412.5 / 425 = 0.971 across the anchor widths while the
    simulator offers 0.400, and the fit wins.
    """
    rate, fixed = _ratio(4, 8, fit={16: (100.0, 400.0)}, origami=(40.0, 100.0), mode="origami")
    assert abs(rate - 412.5 / 425.0) < 1e-9
    assert rate == fixed
    assert "Raised to the 0.971" in capsys.readouterr().out


def test_the_floor_does_not_bind_when_the_simulator_is_already_above_it():
    """Which is the case on every model measured at two widths so far, so this
    guard changes nothing today and is only there to catch the inverse."""
    rate, _fixed = _ratio(4, 8, fit={16: (100.0, 0.0)}, origami=(76.8, 100.0), mode="origami")
    assert abs(rate - 0.768) < 1e-9


def _curve(fixed, per_tok):
    return {"fixed_ms": fixed, "ms_per_token": per_tok, "ms_per_token_sq": 0.0}


def test_the_floor_is_read_at_the_length_being_asked_about(capsys):
    """A single number for the measured ratio is right at one prompt length and
    wrong at every other, because a short step is mostly the cost that does not
    shard.

    These are Llama-3.1-8B's measured TP2 and TP4 intercepts and slopes, with
    the quadratic term dropped so the arithmetic is checkable by hand. They
    carry a step by 0.761 at 1024 tokens and 0.685 at 8192, and the fixture
    asks about 8192, so 0.685 is what a simulator claiming 0.400 is raised to.
    """
    rate, fixed = _ratio(
        4,
        8,
        origami=(40.0, 100.0),
        mode="origami",
        curves={2: _curve(15.20, 0.01328), 4: _curve(11.53, 0.00896)},
    )
    # asked at 8192 by the fixture's request_config
    assert abs(rate - 84.93032 / 123.98976) < 1e-9
    assert rate == fixed
    assert "TP2 and TP4 measured for a 8192-token step" in capsys.readouterr().out


def test_a_width_jump_of_a_different_size_is_not_compared():
    """TP1 -> TP4 says nothing about TP4 -> TP8: four-fold sharding is not two
    applications of two-fold, so the floor declines rather than approximating."""
    rate, _fixed = _ratio(
        4,
        8,
        origami=(40.0, 100.0),
        mode="origami",
        curves={1: _curve(7.34, 0.01701), 4: _curve(11.53, 0.00896)},
    )
    assert abs(rate - 0.400) < 1e-9


def test_the_floor_is_read_at_the_width_of_the_step_not_the_request():
    """Sixteen 1024-token prompts in one step is a 16384-token step, and the
    measured ratio there is nothing like the ratio for a lone 1024-token prompt.

    Reading it at the request length instead lifted GLM-5.2-MXFP4's ISL-1024
    TTFT to +56.0% at 128 concurrent, having improved it to +0.8% at 8.
    """
    p = InferencePerformanceProjector.__new__(InferencePerformanceProjector)
    p.cfg = SimpleNamespace(
        request_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_concurrency=128, batch_size=0
        )
    )
    assert p._step_tokens(1024) == 16384  # budget-bound
    assert p._step_tokens(8192) == 16384  # two prompts fill it
    p.cfg.request_config.max_concurrency = 2
    assert p._step_tokens(1024) == 2048  # only two requests exist
    p.cfg.request_config.max_concurrency = 0
    assert p._step_tokens(1024) == 16384  # unknown concurrency: assume full
