###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A speculative benchmark and a plain one do not measure the same thing.

A plain decode point is one forward pass emitting one token. A speculative one
is a VERIFY step: ``q_len`` query positions and the draft head, in a single
pass. The projector builds the second out of the first by multiplying by
``q_len``, so handed a measurement that already spans ``q_len`` it charges for
it twice -- and because a decode step reads the weights and the KV once for all
of those positions, the multiplication is itself an over-charge.

Neither failure shows up as an error; both come back as a plausible number
several-fold off. So which observable an artifact carries is decided from its
metadata, and every mismatch is refused rather than approximated.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection.benchmark_vllm import (
    _lift_speculative_args,
)
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector as Proj,
)

# EAGLE as the GLM-5.2 sweep runs it: 6 drafted tokens, so a verify step spans 7
# query positions, and a draft pass costing 5% of one target decode token.
DRAFTED = 6
VERIFY_Q_LEN = DRAFTED + 1
PER_TOKEN_MS = 10.0


def projector(*, drafted: int = DRAFTED, restore: bool = False) -> Proj:
    """A projector carrying only the state these rules read.

    ``__new__`` because the real constructor builds the whole profiler tree and
    needs the GEMM simulation backend; none of that is involved in deciding what
    an artifact measured.
    """
    proj = Proj.__new__(Proj)
    proj._meas_verify_q_len = 0
    proj._restore = restore
    proj.cfg = SimpleNamespace(request_config=SimpleNamespace(
        speculative_num_tokens=drafted, speculative_draft_cost_factor=0.05))
    return proj


def spec_meta(**over) -> dict:
    meta = {"speculative_method": "eagle",
            "speculative_num_tokens": DRAFTED,
            "simulate_acc_len": 1.0}
    meta.update(over)
    return meta


POINTS = [(8, PER_TOKEN_MS), (16, 2 * PER_TOKEN_MS)]


# --- which observable the artifact carries ----------------------------------

def test_a_verify_step_anchor_is_consumed_as_the_step():
    proj = projector()
    assert proj._usable_decode_points(POINTS, spec_meta()) == POINTS
    assert proj._meas_verify_q_len == VERIFY_Q_LEN
    # Nothing is synthesised on top: the measurement already spans the query
    # positions, and the draft head ran inside it.
    assert proj._measured_decode_span_ms(PER_TOKEN_MS, VERIFY_Q_LEN) == PER_TOKEN_MS
    assert proj._draft_overhead_ms(PER_TOKEN_MS) == 0.0


def test_a_single_token_anchor_still_has_the_verify_step_built_from_it():
    """The legacy path, unchanged -- and the reason the two must be told apart."""
    proj = projector()
    assert proj._usable_decode_points(POINTS, {}) == POINTS
    assert proj._meas_verify_q_len == 0
    assert proj._measured_decode_span_ms(PER_TOKEN_MS, VERIFY_Q_LEN) == (
        PER_TOKEN_MS * VERIFY_Q_LEN
    )
    # 0.05 of a target decode token per drafted token.
    assert proj._draft_overhead_ms(PER_TOKEN_MS) == pytest.approx(
        0.05 * DRAFTED * PER_TOKEN_MS
    )


def test_a_non_speculative_target_is_untouched_by_any_of_this():
    proj = projector(drafted=0)
    assert proj._usable_decode_points(POINTS, {}) == POINTS
    assert proj._meas_verify_q_len == 0
    assert proj._measured_decode_span_ms(PER_TOKEN_MS, 1) == PER_TOKEN_MS
    assert proj._draft_overhead_ms(PER_TOKEN_MS) == 0.0


# --- what is refused --------------------------------------------------------
# Refusal drops the decode curve, which leaves decode on the simulator. That is
# a worse projection than a good anchor and a better one than a wrong anchor.

def test_an_unforced_acceptance_length_is_refused():
    """Anything but 1.0 leaves an acceptance rate inside the measurement.

    Its value would then come from the benchmark's random prompts, where a draft
    head predicts almost nothing -- an acceptance no real corpus has, welded
    into the artifact.
    """
    proj = projector()
    assert proj._usable_decode_points(POINTS, spec_meta(simulate_acc_len=3.61)) == []
    assert proj._meas_verify_q_len == 0
    assert proj._usable_decode_points(POINTS, spec_meta(simulate_acc_len=None)) == []


def test_a_different_verify_width_is_refused():
    """A 6-position step and a 7-position one are different steps."""
    proj = projector()
    assert proj._usable_decode_points(POINTS, spec_meta(
        speculative_num_tokens=DRAFTED - 1)) == []
    assert proj._meas_verify_q_len == 0


def test_a_speculative_anchor_is_refused_across_parallelism():
    """The restore prices a decode step at one token (``_origami_steps``), so it
    would move a verify step by a single-token delta."""
    proj = projector(restore=True)
    assert proj._usable_decode_points(POINTS, spec_meta()) == []
    assert proj._meas_verify_q_len == 0


def test_refusal_leaves_a_plain_anchor_alone():
    """Only the speculative reading is gated; a restored plain anchor is the
    established path and keeps working."""
    proj = projector(drafted=0, restore=True)
    assert proj._usable_decode_points(POINTS, {}) == POINTS


# --- reaching the artifact at all -------------------------------------------

def test_speculation_is_lifted_out_of_the_engine_flag_string():
    """It is an engine flag on the serving path, and the regime axes and the
    artifact cannot parse one."""
    args = Namespace(
        speculative_method=None, speculative_num_tokens=None,
        server_args="--speculative-algorithm EAGLE3 "
                    f"--speculative-num-draft-tokens {VERIFY_Q_LEN}",
    )
    _lift_speculative_args(args)
    assert args.speculative_method == "EAGLE3"
    # SGLang's flag counts the bonus token; these axes count drafted tokens.
    assert args.speculative_num_tokens == DRAFTED


def test_lifting_never_overrides_what_the_caller_said():
    args = Namespace(speculative_method="ngram", speculative_num_tokens=3,
                     server_args="--speculative-algorithm EAGLE3 "
                                 "--speculative-num-draft-tokens 9")
    _lift_speculative_args(args)
    assert (args.speculative_method, args.speculative_num_tokens) == ("ngram", 3)


def test_a_plain_launch_stays_plain():
    args = Namespace(speculative_method=None, speculative_num_tokens=None,
                     server_args="--attention-backend triton")
    _lift_speculative_args(args)
    assert args.speculative_method is None
    assert args.speculative_num_tokens is None
