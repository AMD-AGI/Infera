###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Per-rank inference memory must reflect what a rank actually holds.

Three separate overestimates each pushed gpt-oss-120b's reported footprint past
an MI355X's 288 GB and drove the projected sustainable concurrency to single
digits: weights were counted un-sharded, ``mxfp4`` silently fell back to bf16,
and the prefill activation working set ignored the scheduler's token budget.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.training_config import dtype_num_bytes

GIB = 1024.0 ** 3


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("mxfp4", 0.53125),   # 4 bits + one E8M0 scale byte per 32 elements
        ("mxfp8", 1.03125),
        ("fp8", 1.0),
        ("bf16", 2.0),
    ],
)
def test_block_scaled_dtypes_are_not_silently_bf16(dtype, expected):
    assert dtype_num_bytes(dtype) == pytest.approx(expected)


def test_mxfp4_is_a_quarter_of_bf16():
    """The whole point of the format; a fallback to 2.0 would hide 4x of HBM."""
    assert dtype_num_bytes("mxfp4") < dtype_num_bytes("bf16") / 3.5


from .conftest import project_spec as _project


def test_weights_are_tp_sharded_across_ranks():
    """Doubling TP halves the per-rank weight footprint."""
    tp4 = _project(tp=4, concurrency=1, input_len=128, output_len=8)
    tp8 = _project(tp=8, concurrency=1, input_len=128, output_len=8)
    ratio = tp4["memory_gb"] / tp8["memory_gb"]
    assert 1.7 < ratio < 2.3, f"TP4/TP8 per-rank memory ratio {ratio:.2f} is not ~2x"


def test_expert_parallelism_does_not_shrink_the_model():
    """Turning EP on redistributes experts across the same GPUs; it deletes none.

    The profiler divides expert weights by EP because in training EP is its own
    axis of GPUs. A serving engine places experts on the tensor-parallel ranks
    instead (vLLM sets EP = TP), so applying that split *and* the TP one charged
    a rank a fraction of the experts it really loads -- reporting a 120B MoE at
    under a gigabyte of weights per rank, and conjuring the HBM to prove it fit.
    """
    off = _project(tp=8, ep=1, concurrency=1, input_len=128, output_len=8)
    on = _project(tp=8, ep=8, concurrency=1, input_len=128, output_len=8)
    assert on["memory_gb"] == pytest.approx(off["memory_gb"], rel=1e-6)
    # Equality alone would also hold if both arms under-counted, which is the
    # failure in question: 116.5B at 0.53 B/param over 8 ranks is ~7 GB a rank.
    assert on["memory_gb"] > 5.0


def test_gpt_oss_mxfp4_fits_on_one_mi355x():
    """116.5B params at 0.53 B/param over TP=8 is ~7 GB of weights per rank."""
    out = _project(tp=8, weight_dtype="mxfp4", concurrency=1, input_len=1024, output_len=8)
    assert out["memory_gb"] < 40.0, f"per-rank total {out['memory_gb']:.1f} GB"
    assert out["sustainable_concurrency"] > 1000


def test_long_context_prefill_respects_the_token_budget():
    """A 128k prompt at batch 64 must not be charged 64x128k live activations."""
    out = _project(
        tp=8,
        concurrency=64,
        input_len=131072,
        output_len=128,
        max_num_batched_tokens=8192,
    )
    assert out["sustainable_concurrency"] > 0, "activation blow-up starved the KV cache"
    assert out["memory_gb"] < 288.0
