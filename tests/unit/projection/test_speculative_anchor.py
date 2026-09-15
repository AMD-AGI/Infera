###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What a speculation-harvested decode measurement means, and what it costs to
forget.

The harvester differences its decode timing with the draft config applied, so
with speculation on it reports a per-*output-token* latency that already has
acceptance folded in -- the same quantity a served TPOT reports. The projector
used to scale that number by the verify width ``k + 1``, as if it were a
single-token step being stretched into a verify pass. That charges the draft
twice: decode inflated by ``(k + 1) / tokens_per_step`` and throughput fell by
the same ratio, so speculation looked worse the better it was accepted, which
is the opposite of what it does.

These are written as invariants rather than expected values, because the honest
statement is a cancellation: the step cost and the step count are the same
quantity seen from two directions, so they have to cancel exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
)


class _Anchored:
    """The two methods under test, over just the state they read.

    Constructing a real projector needs a model preset, a parallel layout and a
    GEMM backend, none of which this arithmetic touches.
    """

    _measured_verify_step_scale = InferencePerformanceProjector._measured_verify_step_scale
    _spec_tokens_per_step = InferencePerformanceProjector._spec_tokens_per_step

    def __init__(self, *, spec_k: int, accept: float, anchor_spec_k: int):
        self.cfg = SimpleNamespace(
            request_config=SimpleNamespace(
                speculative_num_tokens=spec_k,
                speculative_acceptance_rate=accept,
            )
        )
        self._bench_spec_k = anchor_spec_k


# (k, acceptance) pairs spanning the useful range. k=1/0.84 is GLM-5.2-MXFP4's
# MTP level 1, whose measured acceptance length is 1.84.
_SPEC_CASES = [(1, 0.84), (1, 0.70), (2, 0.80), (3, 0.70), (3, 0.90), (5, 0.80)]


@pytest.mark.parametrize("spec_k,accept", _SPEC_CASES)
def test_a_spec_harvested_step_scales_by_tokens_emitted_not_verify_width(spec_k, accept):
    """The step scale has to cancel the step count exactly.

    ``decode_total_ms`` divides the output length by the tokens a step emits, so
    for the total to come back to ``decode_ms x output_len`` -- which is what a
    per-output-token measurement already says it is -- the step has to cost that
    same factor. Any other scale leaves a residue.
    """
    p = _Anchored(spec_k=spec_k, accept=accept, anchor_spec_k=spec_k)
    scale = p._measured_verify_step_scale()
    emitted = p._spec_tokens_per_step()

    assert scale == pytest.approx(emitted), (
        f"k={spec_k} accept={accept}: step scaled by {scale:.4f} against "
        f"{emitted:.4f} tokens emitted; the residue lands in decode time"
    )


@pytest.mark.parametrize("spec_k,accept", _SPEC_CASES)
def test_decode_time_and_throughput_stop_depending_on_draft_depth(spec_k, accept):
    """Per-sequence decode rate must be the anchor's own rate.

    A per-output-token anchor already answers "how long is a token", so no draft
    depth or acceptance rate may move it. Before the fix this read
    ``1000 x emitted / (decode_ms x (k+1))`` and speculation was reported as a
    throughput loss.
    """
    decode_ms = 4.0  # ms per output token, acceptance already folded in
    p = _Anchored(spec_k=spec_k, accept=accept, anchor_spec_k=spec_k)

    step_ms = decode_ms * p._measured_verify_step_scale()
    per_seq_tps = 1000.0 * p._spec_tokens_per_step() / step_ms
    assert per_seq_tps == pytest.approx(1000.0 / decode_ms), (
        f"k={spec_k} accept={accept}: {per_seq_tps:.2f} tok/s against the "
        f"anchor's own {1000.0 / decode_ms:.2f}"
    )

    # And the defect is really gone: the old width scaling was strictly worse
    # for every acceptance below perfect.
    if accept < 1.0:
        old_step_ms = decode_ms * (spec_k + 1)
        assert old_step_ms > step_ms, "width scaling should have overcharged the step"
        old_tps = 1000.0 * p._spec_tokens_per_step() / old_step_ms
        assert old_tps < per_seq_tps


def test_perfect_acceptance_is_where_the_two_scales_agree():
    """At acceptance 1 every draft token lands, so emitted == verify width.

    This is why the defect hid: the one case where width scaling is right is the
    one a quick sanity check reaches for.
    """
    p = _Anchored(spec_k=3, accept=1.0, anchor_spec_k=3)
    assert p._measured_verify_step_scale() == pytest.approx(4.0)
    assert p._spec_tokens_per_step() == pytest.approx(4.0)


@pytest.mark.parametrize("spec_k,accept", _SPEC_CASES)
def test_an_anchor_without_speculation_keeps_the_width_scaling(spec_k, accept):
    """Nothing was folded into a non-speculative measurement, so there is
    nothing to undo.

    A single-token step cannot be turned into a verify pass by arithmetic --
    which is why speculation is regime-defining and the anchor store refuses
    this pairing. ``--load-benchmark`` does not refuse it, so the old behaviour
    stays rather than returning a number that would look calibrated.
    """
    p = _Anchored(spec_k=spec_k, accept=accept, anchor_spec_k=0)
    assert p._measured_verify_step_scale() == pytest.approx(float(spec_k + 1))


def test_no_speculation_leaves_the_measured_step_alone():
    p = _Anchored(spec_k=0, accept=0.0, anchor_spec_k=0)
    assert p._measured_verify_step_scale() == pytest.approx(1.0)
    # An anchor that measured speculation cannot rescale a target that does not
    # ask for it either.
    q = _Anchored(spec_k=0, accept=0.0, anchor_spec_k=3)
    assert q._measured_verify_step_scale() == pytest.approx(1.0)


def test_the_anchor_records_the_draft_depth_it_measured():
    """The fix is only reachable if the artifact says what it measured, so the
    harvester's key and the projector's reader have to stay spelled the same."""
    from infera.projection.core.projection.inference_projection import benchmark_vllm

    src = benchmark_vllm.__file__
    with open(src) as fh:
        assert '"speculative_num_tokens"' in fh.read(), (
            "harvest must keep recording the draft depth in anchor meta"
        )
