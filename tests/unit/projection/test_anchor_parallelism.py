###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Attention-DP is not a parallelism the projector can convert between.

Plain tensor parallelism gives every rank a slice of the attention heads over
the whole batch. Attention-DP gives it every head over a slice of the batch.
Those are different kernel shapes and a different per-rank KV layout, and
nothing in the projector converts one into the other.

What makes it worse than the TP/EP/PP axes is that the switch leaves all three
of those integers untouched -- the sweep's ``TP`` and ``TP+DPA`` modes are both
(tp=width, ep=1, pp=1). So the parallelism restore never engages, and without
the check below the measured curve would be reused verbatim for a strategy it
never ran.
"""

from __future__ import annotations

from types import SimpleNamespace

from infera.projection.core.projection.inference_projection.benchmark_vllm import (
    _lifted_attention_dp,
)
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector as Proj,
)


def projector(target_dp: int) -> Proj:
    """A projector carrying only the target attention-DP degree.

    ``__new__`` because the real constructor builds the whole profiler tree and
    needs the GEMM simulation backend, which has no part in this comparison.
    """
    proj = Proj.__new__(Proj)
    proj.cfg = SimpleNamespace(model_parallel_config=SimpleNamespace(
        attention_data_parallel_size=target_dp))
    return proj


# --- comparing the artifact against the target ------------------------------

def test_a_matching_strategy_is_accepted():
    assert projector(8)._attention_dp_refusal({"attention_dp": 8}) is None
    assert projector(1)._attention_dp_refusal({"attention_dp": 1}) is None


def test_a_plain_tp_anchor_is_refused_for_an_attention_dp_target():
    refusal = projector(8)._attention_dp_refusal({"attention_dp": 1})
    assert refusal and "attention-DP 1" in refusal and "target uses 8" in refusal


def test_an_attention_dp_anchor_is_refused_for_a_plain_tp_target():
    """The direction the restore is blindest to: TP, EP and PP all match, so
    nothing else in the pipeline would look at this."""
    assert projector(1)._attention_dp_refusal({"attention_dp": 8}) is not None


def test_a_different_attention_dp_width_is_refused():
    assert projector(8)._attention_dp_refusal({"attention_dp": 4}) is not None


def test_an_artifact_predating_the_field_is_not_refused():
    """Anchors already on disk keep working; only DPA targets are at risk, and
    those get a warning rather than a refusal."""
    assert projector(1)._attention_dp_refusal({}) is None
    assert projector(8)._attention_dp_refusal({}) is None


# --- reading the degree off the engine flags --------------------------------
# SGLang's own rule is ``dp_size if enable_dp_attention else 1``, so the two
# flags are only meaningful together.

def test_both_flags_together_give_the_degree():
    assert _lifted_attention_dp("--dp-size 8 --enable-dp-attention") == 8
    assert _lifted_attention_dp("--enable-dp-attention --data-parallel-size 4") == 4
    assert _lifted_attention_dp("--tp 8 --dp-size=8 --enable-dp-attention") == 8


def test_the_switch_alone_does_nothing():
    """The pit this is here to catch: attention-DP asked for but never engaged,
    so the run measures plain TP while claiming to measure DPA."""
    assert _lifted_attention_dp("--enable-dp-attention") == 1
    assert _lifted_attention_dp("--enable-dp-attention --dp-size 1") == 1


def test_plain_data_parallelism_is_not_attention_dp():
    assert _lifted_attention_dp("--dp-size 8") == 1


def test_a_launch_without_either_flag_is_one():
    assert _lifted_attention_dp("") == 1
    assert _lifted_attention_dp("--tp 8 --attention-backend aiter") == 1
