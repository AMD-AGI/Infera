###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Hybrid-model layers priced from their own operations.

Kimi-K3 and Qwen3.5 run most layers as linear attention (KDA / gated delta
net), and Kimi-K3 adds an MLA output gate, a latent MoE and attention
residuals. Each is costed from the shapes in the published modelling code
rather than as an attention layer over ``d`` keys.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection.kv_cache import (
    linear_state_bytes_per_layer,
)
from infera.projection.core.projection.module_profilers.linear_attention import (
    LinearAttentionProfiler,
)
from infera.projection.core.projection.training_config import (
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
    decode_kernels_per_layer,
)

from .conftest import project_spec

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
    linear_attention_kind="kda",
    linear_attention_full_rank_gate=True,
)

_QWEN35 = dict(
    num_layers=60,
    hidden_size=4096,
    num_attention_heads=32,
    kv_channels=256,
    linear_attention_freq=4,
    linear_attention_conv_kernel=4,
    linear_attention_kind="gdn",
    linear_num_key_heads=16,
    linear_num_value_heads=64,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
)


class _WeightGemm:
    """A GEMM whose time is its weight elements, in ms, and records its shapes."""

    hbm_bandwidth_gbps = 1e15

    def __init__(self, scale=1.0):
        self.calls = []
        self.scale = scale

    def simulate_gemm(self, m, n, k, dtype="bf16", batch=1):
        self.calls.append((m, n, k, batch))
        return SimpleNamespace(forward_time_ms=self.scale * n * k * batch)


def _view(mc, tp=1):
    return SimpleNamespace(
        model_config=ModelConfig(**mc),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=tp),
    )


def test_kda_decode_reads_every_projection_weight_once():
    """q, k, v, the full gate and the output are 7168 x 96*128 each; the decay gate is low rank."""
    gemm = _WeightGemm()
    LinearAttentionProfiler(_view(_KIMI, tp=8), gemm).layer_ms(1, 1, "decode")
    h, p, d, heads = 7168, 96 * 128 // 8, 128, 96 // 8
    expected = 5 * h * p + h * d + d * p + h * heads
    assert sum(n * k for _, n, k, _ in gemm.calls) == expected
    assert len(gemm.calls) == 8


def test_kda_layer_weights_are_about_twice_an_mla_layer():
    gemm = _WeightGemm()
    LinearAttentionProfiler(_view(_KIMI), gemm).layer_ms(1, 1, "decode")
    kda = sum(n * k for _, n, k, _ in gemm.calls)
    mla = 7168 * 1536 + 1536 * 96 * 192 + 7168 * 512 + 512 * 96 * 256 + 7168 * 64 + 96 * 128 * 7168
    gate = 7168 * 96 * 128
    assert kda == pytest.approx(443e6, rel=0.01)
    assert 1.8 < kda / (mla + gate) < 2.0


def test_decode_state_read_grows_with_the_batch():
    """Every sequence's dk x dv state per value head is read and rewritten each step."""
    gemm = _WeightGemm(scale=0.0)
    view = _view(_KIMI, tp=8)
    prof = LinearAttentionProfiler(view, gemm, hbm_bandwidth_gbps=1000.0)
    per_seq = prof.layer_ms(65, 1, "decode") - prof.layer_ms(1, 1, "decode")
    state_bytes = 2 * (96 // 8) * 128 * 128 * 2
    floor = 64 * state_bytes / (1000.0 * 0.566 * 1e6)
    assert floor < per_seq < 1.2 * floor


def test_gdn_runs_two_fused_projections_and_the_output():
    gemm = _WeightGemm()
    LinearAttentionProfiler(_view(_QWEN35), gemm).layer_ms(4, 1, "decode")
    shapes = [(n, k) for _, n, k, _ in gemm.calls]
    assert shapes == [(2 * 16 * 128 + 2 * 64 * 128, 4096), (2 * 64, 4096), (4096, 64 * 128)]


def test_prefill_adds_the_chunked_recurrence():
    gemm = _WeightGemm()
    prof = LinearAttentionProfiler(_view(_QWEN35), gemm)
    prof.layer_ms(1, 4096, "prefill")
    batched = [c for c in gemm.calls if c[3] > 1]
    assert len(batched) == 6
    assert all(b == 64 * 4096 // 64 for *_, b in batched)


def test_linear_state_is_one_matrix_per_value_head():
    """Qwen3.5 has 64 value heads over 16 key heads; the state follows the value heads."""
    cfg = InferenceConfig(
        model_config=ModelConfig(**_QWEN35),
        request_config=InferenceRequestConfig(),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=1),
    )
    conv = (2 * 16 * 128 + 64 * 128) * 3 * 2
    assert linear_state_bytes_per_layer(cfg) == 64 * 128 * 128 * 2 + conv


def test_kernel_count_follows_the_layer_types():
    plain = ModelConfig(**{**_KIMI, "linear_attention_kind": ""})
    priced = ModelConfig(**_KIMI)
    # MLA 9 on 24 layers, KDA 13 on 69: (24*9 + 69*13) / 93 = 11.97 attention kernels
    assert decode_kernels_per_layer(plain) == 4 + 9 + 3
    assert decode_kernels_per_layer(priced) == round(4 + (24 * 9 + 69 * 13) / 93 + 3)
    gated = ModelConfig(
        **{
            **_KIMI,
            "attention_output_gate": True,
            "num_experts": 896,
            "moe_latent_hidden_size": 3584,
            "attn_res_block_size": 12,
        }
    )
    assert decode_kernels_per_layer(gated) == round(4 + (24 * 11 + 69 * 13) / 93 + 7 + 3 + 2)


def test_kimi_decode_is_slower_once_kda_is_priced(monkeypatch):
    common = dict(
        model="kimi_k3",
        tp=8,
        ep=8,
        concurrency=16,
        input_len=8192,
        output_len=64,
        weight_dtype="mxfp4",
        kv_cache_dtype="fp8",
    )
    priced = project_spec(**common)
    monkeypatch.setenv("INFERASIM_LINEAR_ATTN_BLEND", "1")
    blended = project_spec(**common)
    assert priced["tpot_ms"] > blended["tpot_ms"]
