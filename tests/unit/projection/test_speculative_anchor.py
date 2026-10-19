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
    """The methods under test, over just the state they read.

    Constructing a real projector needs a model preset, a parallel layout and a
    GEMM backend, none of which this arithmetic touches.

    ``step_ms`` stands in for the analytical model the width ratio is taken
    from: a mapping from query width to what a step of that width costs. Left
    unset, the model is unreachable -- which is the case on a host without the
    simulator, and the case the blind fallback exists for.
    """

    _measured_verify_step_scale = InferencePerformanceProjector._measured_verify_step_scale
    _spec_tokens_per_step = InferencePerformanceProjector._spec_tokens_per_step
    _verify_width_ratio = InferencePerformanceProjector._verify_width_ratio

    def __init__(
        self,
        *,
        spec_k: int,
        accept: float,
        anchor_spec_k: int,
        step_ms: dict[int, float] | None = None,
    ):
        self.cfg = SimpleNamespace(
            request_config=SimpleNamespace(
                speculative_num_tokens=spec_k,
                speculative_acceptance_rate=accept,
                input_seq_len=8192,
            )
        )
        self._bench_spec_k = anchor_spec_k
        self._meas_ref_input = 8192
        self._verify_ratio_cache: dict = {}
        self._step_ms = step_ms

    def _forward_times(self, batch, q_len, phase, kv_len):
        if self._step_ms is None:
            raise RuntimeError("analytical backend unavailable on this host")
        return SimpleNamespace(total_ms=self._step_ms[q_len])


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
def test_a_non_speculative_anchor_falls_back_to_width_with_no_model(spec_k, accept):
    """Nothing was folded into a non-speculative measurement, so there is
    nothing to undo, and without a model to price the widening there is
    nothing better than the width to charge.

    A single-token step cannot be turned into a verify pass by arithmetic --
    which is why speculation is regime-defining and the anchor store refuses
    this pairing. ``--load-benchmark`` does not refuse it, so the blind width
    stands rather than returning a number that would look calibrated.
    """
    p = _Anchored(spec_k=spec_k, accept=accept, anchor_spec_k=0)
    assert p._measured_verify_step_scale() == pytest.approx(float(spec_k + 1))


@pytest.mark.parametrize("spec_k,accept", _SPEC_CASES)
def test_a_non_speculative_anchor_prices_the_widening_from_the_model(spec_k, accept):
    """With a model reachable, the verify pass costs what it costs.

    The width is an upper bound, not an estimate: a verify pass puts ``k + 1``
    query positions through the stack but reads the KV cache once for the same
    sequences, and at the contexts these anchors are read at that read is most
    of the step. Charging the width made a target read slower the deeper its
    draft, which is backwards.
    """
    width = spec_k + 1
    # A step that widens by a tenth per extra position: memory-bound decode,
    # where the query dimension is nearly free.
    steps = {1: 10.0, width: 10.0 * (1.0 + 0.1 * spec_k)}
    p = _Anchored(spec_k=spec_k, accept=accept, anchor_spec_k=0, step_ms=steps)
    scale = p._measured_verify_step_scale()

    assert scale == pytest.approx(1.0 + 0.1 * spec_k)
    assert scale < width, "the modelled widening must undercut the blind width"


@pytest.mark.parametrize("wide_ms,expected", [(5.0, 1.0), (1000.0, 4.0)])
def test_the_modelled_widening_stays_between_one_step_and_one_per_position(wide_ms, expected):
    """A verify pass cannot beat the single-token step inside it, and cannot
    cost more than running that step once per position. A model that says
    otherwise is answering about a different shape than the one being priced.
    """
    p = _Anchored(spec_k=3, accept=0.8, anchor_spec_k=0, step_ms={1: 10.0, 4: wide_ms})
    assert p._measured_verify_step_scale() == pytest.approx(expected)


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
