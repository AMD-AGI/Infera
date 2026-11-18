###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A short prefill chunk pays a launch, not just a per-token rate.

Measured on Qwen2.5-1.5B, one MI325, radix off: 512 tokens took 8.9 ms and
2048 tokens took 14.6 ms. A single rate fitted at 2048 bills the 512-token
chunk at about 3 ms and the chunk-size ranking comes out backwards. Two
measured lengths keep the floor. A single-length anchor is unchanged.

A mixed step is one forward. The decode sequences in it are extra query
tokens. A one-token leftover still pays the prefill launch: 32 tokens and
512 tokens measured the same step time.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from infera.projection.core.config.preset_loader import PresetLoader
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
    _interp_prefill_token_ms,
)
from infera.projection.core.projection.training_config import (
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
    megatron_derive_default_args,
    update_config_from_args,
)

_SHORT = 8.86
_LONG = 14.61


def _anchor(by_input_len=None):
    blob = {
        "backend": "sglang",
        "measured": {"model": {"prefill_ms": _LONG, "decode_ms": 2.27}},
        "sweep": [
            {"batch": 1, "prefill_ms": _LONG, "decode_ms": 2.24},
            {"batch": 8, "prefill_ms": 91.78, "decode_ms": 2.27},
        ],
        "meta": {
            "batch": 1,
            "input_len": 2048,
            "tp": 1,
            "benchmark_tp": 1,
            "prefix_caching": False,
        },
    }
    if by_input_len is not None:
        blob["by_input_len"] = by_input_len
    return blob


def _two_lengths():
    return [
        {"batch": 1, "input_len": 512, "prefill_ms": _SHORT},
        {"batch": 1, "input_len": 2048, "prefill_ms": _LONG},
    ]


@pytest.fixture(scope="module")
def cfg():
    preset = PresetLoader.load("qwen2.5_3B", framework="megatron")
    preset.update(
        hidden_size=1536,
        ffn_hidden_size=8960,
        num_layers=28,
        num_attention_heads=12,
        num_query_groups=2,
        group_query_attention=True,
        kv_channels=128,
        padded_vocab_size=151936,
        max_position_embeddings=32768,
        untie_embeddings_and_output_weights=False,
        rotary_base=1000000,
        norm_epsilon=1.0e-6,
        seq_length=2048,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        num_experts=None,
        moe_layer_freq=1,
    )
    args = SimpleNamespace(**preset)
    megatron_derive_default_args(args)
    model = update_config_from_args(ModelConfig(), args)
    req = InferenceRequestConfig(
        input_seq_len=2048,
        output_seq_len=47,
        batch_size=8,
        max_concurrency=8,
        chunked_prefill_size=2048,
        max_num_batched_tokens=2048,
        weight_dtype="bf16",
        kv_cache_dtype="bf16",
        serving_model="continuous",
        prefix_cache_hit_rate=0.0,
    )
    return InferenceConfig(
        model_config=model,
        request_config=req,
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=1),
    )


def _project(cfg, blob):
    gpu = Namespace(gpu_arch="mi325x", gpu_clock_mhz=None, gemm_backend=None)
    return InferencePerformanceProjector(cfg, args=gpu, benchmark_layer_times=blob)


def test_below_the_shortest_point_the_line_does_not_pass_through_zero():
    curve = [(512.0, _SHORT), (2048.0, _LONG)]
    assert _interp_prefill_token_ms(512, curve) == pytest.approx(_SHORT)
    assert _interp_prefill_token_ms(2048, curve) == pytest.approx(_LONG)
    # 64 tokens at the 2048 rate would be ~0.5 ms. The floor keeps it up near
    # the 512-token measurement.
    assert _interp_prefill_token_ms(64, curve) > 5.0


def test_a_single_length_anchor_stays_a_rate(cfg):
    projector = _project(cfg, _anchor())
    assert projector._meas_prefill_token_curve == []
    # The single-length rate is the mean across the batch sweep, at the
    # anchor's prompt length. No floor.
    rate = (_LONG / 2048.0 + 91.78 / (8 * 2048.0)) / 2.0
    assert projector._measured_prefill_tokens_ms(512) == pytest.approx(rate * 512, rel=1e-3)


def test_two_lengths_bill_the_short_chunk_at_the_measurement(cfg):
    projector = _project(cfg, _anchor(by_input_len=_two_lengths()))
    assert projector._measured_prefill_tokens_ms(512) == pytest.approx(_SHORT, rel=1e-3)
    # The single-rate bill for this chunk is about 3.6 ms. The floor is the point.
    assert projector._measured_prefill_tokens_ms(512) > 8.0


def test_a_mixed_step_is_one_forward(cfg):
    projector = _project(cfg, _anchor(by_input_len=_two_lengths()))
    mixed = projector.mixed_step_latency_ms(4, 2048, 2048, 2048)
    one_forward = projector._measured_prefill_tokens_ms(2048 + 4)
    decode = projector.decode_step_latency_ms(4, 2048)
    assert mixed == pytest.approx(one_forward, rel=1e-6)
    assert mixed < one_forward + 0.5 * decode


def test_a_one_token_prefill_still_pays_the_launch(cfg):
    """32 tokens measured the same as 512. A leftover does not become a decode."""
    projector = _project(cfg, _anchor(by_input_len=_two_lengths()))
    mixed = projector.mixed_step_latency_ms(8, 1, 2048, 2048)
    decode = projector.decode_step_latency_ms(8, 2048)
    assert mixed == pytest.approx(projector._measured_prefill_tokens_ms(1 + 8), rel=1e-6)
    assert mixed > 2.0 * decode
