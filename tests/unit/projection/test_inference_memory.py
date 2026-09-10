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

from .conftest import project_spec as _project

GIB = 1024.0**3


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("mxfp4", 0.53125),  # 4 bits + one E8M0 scale byte per 32 elements
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


def test_deepseek_v4_kv_is_the_576_byte_shared_latent():
    """V4 caches one latent per token, not K and V of kv_channels each.

    The yaml leaves ``multi_latent_attention`` off for the trainer. The serving
    cache is still the 576-byte layout vLLM registers (512-d shared K=V + 64-d
    RoPE), which is what lets a TP8 decode replica hold ~4k sequences of 8k.
    """
    from types import SimpleNamespace

    from infera.projection.core.projection.inference_projection.kv_cache import (
        kv_bytes_per_token_per_layer,
    )

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            multi_latent_attention=False,
            kv_lora_rank=0,
            qk_pos_emb_head_dim=64,
            group_query_attention=True,
            num_query_groups=1,
            num_attention_heads=128,
            kv_channels=512,
        ),
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=8,
            attention_data_parallel_size=1,
        ),
        request_config=SimpleNamespace(kv_cache_dtype="fp8"),
    )
    assert kv_bytes_per_token_per_layer(cfg) == pytest.approx(576.0)


def test_disagg_kv_is_split_across_decode_replicas():
    """Each decode replica holds C / replicas sequences, not the system batch.

    Without the split, a 2-decode-replica fleet at concurrency 256 was charged
    the same KV as a 1-replica fleet at 256, so doubling the decode pool did
    not free HBM and the measured MiniMax GB300 4xP/2xD winner at 4096 missed
    the 288 GB ceiling by a replica-count.
    """
    common = dict(
        disaggregate=True,
        prefill_tp=4,
        tp=8,
        ep=1,
        prefill_replicas=1,
        concurrency=256,
        input_len=1024,
        output_len=128,
        max_num_batched_tokens=8192,
    )
    one = _project(**common, decode_replicas=1)
    two = _project(**common, decode_replicas=2)
    assert two["kv_cache_gb"] < 0.6 * one["kv_cache_gb"], (
        f"2 decode replicas still hold {two['kv_cache_gb']:.1f} GB of KV vs "
        f"{one['kv_cache_gb']:.1f} GB on 1 replica at the same system C"
    )
    twice_common = {**common, "decode_replicas": 2, "concurrency": 512}
    twice = _project(**twice_common)
    assert twice["kv_cache_gb"] == pytest.approx(one["kv_cache_gb"], rel=0.15)


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
