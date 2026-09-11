###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Hybrid linear attention (KDA / GDN) is architecture, not a cache-manager option.

Kimi-K3 runs 69 of 93 layers as Kimi Delta Attention: a d×d recurrent state
per head, not a per-token KV cache. Charging those layers as MLA oversizes
the cache ~3.9x and makes prefill attention grow with context on layers that
do not. Sliding-window is a different fact -- the engine may not deliver it
at decode -- so these asserts live here rather than in the window tests.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.kv_cache import (
    estimate_kv_cache,
)
from infera.projection.core.projection.training_config import (
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
)

from .conftest import project_spec

# Kimi-K3 architecture, read off the checkpoint: 24 full MLA + 69 KDA.
_KIMI = dict(
    num_layers=93,
    hidden_size=7168,
    num_attention_heads=96,
    kv_channels=128,
    multi_latent_attention=True,
    kv_lora_rank=512,
    qk_pos_emb_head_dim=64,
    linear_attention_layers=69,
    linear_attention_head_dim=128,
    linear_attention_conv_kernel=4,
)


def _kimi_cfg(**req):
    return InferenceConfig(
        model_config=ModelConfig(**_KIMI),
        request_config=InferenceRequestConfig(kv_cache_dtype="fp8", **req),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=8),
    )


def test_kda_state_is_the_head_dim_not_a_fitted_window():
    """A d×d state per head is attending to d keys. d comes from the yaml."""
    mc = ModelConfig(**_KIMI)
    assert mc.linear_attention_layer_count() == 69
    assert mc.linear_attention_state_len() == 128
    assert mc.full_attention_layer_fraction() == pytest.approx(24 / 93)


def test_linear_blend_at_long_context_is_the_mla_fraction():
    """At 130k the average layer reads ~24/93 of the prompt, plus a 128-token state."""
    mc = ModelConfig(**_KIMI)
    kv = 130_000
    blended = mc.blend_linear_attn_kv(kv)
    expected = (69 * 128 + 24 * kv) / 93
    assert blended == pytest.approx(expected, rel=1e-3)
    assert blended < 0.30 * kv, f"blended KV {blended} is not the MLA-layer fraction of {kv}"


def test_linear_blend_is_a_no_op_without_kda_layers():
    """gpt-oss and DeepSeek must not pick up a phantom linear fraction."""
    mc = ModelConfig(num_layers=36, num_attention_heads=64, kv_channels=64)
    assert mc.blend_linear_attn_kv(65536) == 65536


def test_qwen_style_freq_counts_full_attention_every_n_layers():
    mc = ModelConfig(num_layers=40, linear_attention_freq=4, kv_channels=128)
    assert mc.linear_attention_layer_count() == 30
    assert mc.full_attention_layer_fraction() == pytest.approx(0.25)


def test_kimi_token_kv_is_only_the_mla_layers():
    """At 64k the cache is the 24 MLA latents, not 93, plus a small recurrent state."""
    ctx, conc = 65536, 8
    kv = estimate_kv_cache(_kimi_cfg(max_concurrency=conc), 93, context_len=ctx)
    latent = 512 + 64  # kv_lora_rank + rope, fp8
    token_bytes = 24 * latent * ctx * conc
    # Recurrent state is bf16, TP-sharded heads: 69 layers × 12 heads × d×d,
    # plus the short-conv leftover of (kernel-1) tokens.
    d, heads_on_rank, kernel = 128, 96 // 8, 4
    state_bytes = 69 * heads_on_rank * (d * d + d * (kernel - 1)) * 2 * conc
    assert kv.bytes_total == pytest.approx(token_bytes + state_bytes, rel=1e-6)
    all_mla = 93 * latent * ctx * conc
    assert kv.bytes_total < 0.35 * all_mla


def test_kimi_yaml_keys_reach_the_projection():
    """If the preset keys never landed on ModelConfig, the cache would be 93-layer MLA."""
    kimi = project_spec(
        model="kimi_k3",
        tp=8,
        ep=8,
        concurrency=8,
        input_len=65536,
        output_len=8,
        weight_dtype="mxfp4",
        kv_cache_dtype="fp8",
    )
    ctx = 65536 + 8
    latent = 512 + 64
    mla_only_gb = 24 * latent * ctx * 8 / (1024.0**3)
    all_mla_gb = 93 * latent * ctx * 8 / (1024.0**3)
    assert kimi["kv_cache_gb"] < 0.40 * all_mla_gb, (
        f"Kimi KV {kimi['kv_cache_gb']:.2f} GB looks like 93-layer MLA "
        f"({all_mla_gb:.2f} GB), so the linear_attention_* keys were dropped"
    )
    assert kimi["kv_cache_gb"] > 0.80 * mla_only_gb


def test_kimi_prefill_does_not_grow_as_dense_mla():
    """Same 8k of new tokens attending 8k vs 64k: KDA layers do not go quadratic.

    Dense 93-layer MLA would grow attention with the prefix. After the blend,
    TTFT can still rise (the 24 MLA layers and the MLP of the new tokens), but
    not by the ~8x of a full-context prefill of 64k vs 8k.
    """
    common = dict(
        model="kimi_k3",
        tp=8,
        ep=8,
        concurrency=1,
        output_len=8,
        weight_dtype="mxfp4",
        kv_cache_dtype="fp8",
    )
    short = project_spec(**common, input_len=8192, prefix_cache_hit_rate=0.0)
    # 65536 * (1 - 0.875) = 8192 new tokens attending 64k.
    long = project_spec(**common, input_len=65536, prefix_cache_hit_rate=0.875)
    growth = long["ttft_ms"] / short["ttft_ms"]
    assert growth < 3.0, (
        f"Kimi TTFT grew {growth:.2f}x for the same 8k suffix on a 64k prefix; "
        f"93-layer MLA would have grown with the prefix on every layer"
    )
    assert long["ttft_ms"] > short["ttft_ms"], (
        "the 24 MLA layers still read the prefix, so long context is not free"
    )
