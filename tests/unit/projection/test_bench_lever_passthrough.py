###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A benchmark that ignores the flags it was asked to run under measures nothing.

The reduced-scale benchmark exists so a serving variant can be screened cheaply:
run the same engine with the same flags on fewer GPUs and fewer layers, and rank
the candidates by what comes back. That only means anything if the variant's own
flags and env actually reach the engine -- and if two runs made under different
flags cannot be mistaken for each other by the anchor store afterwards.
"""

from __future__ import annotations

import argparse

import pytest

from infera.projection.core.projection.inference_projection import benchmark_vllm
from infera.projection.core.projection.inference_projection.search.regime import (
    recipe_from_bench_args,
    regime_distance,
    regime_signature,
)


def bench_args(**over):
    defaults = dict(
        model="openai/gpt-oss-120b", tp=8, pp=1, enable_expert_parallel=False,
        quantization=None, kv_cache_dtype=None, enforce_eager=False,
        num_hidden_layers=None, batch=32, input_len=1024, output_len=1024,
        speculative_method=None, speculative_num_tokens=None, no_aiter=False,
    )
    defaults.update(over)
    return argparse.Namespace(**defaults)


# --- the levers reach the engine -------------------------------------------

def test_empty_server_args_change_nothing():
    assert benchmark_vllm._engine_kwargs_from_server_args("") == {}
    assert benchmark_vllm._engine_kwargs_from_server_args(None) == {}


def test_env_overrides_are_applied_before_vllm_is_imported(monkeypatch):
    """The ROCm levers are read once at import, so ordering is the whole point."""
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args(["--env", "VLLM_ROCM_USE_AITER=0",
                              "--env", "VLLM_ATTENTION_BACKEND=TRITON_ATTN"])
    for item in sorted(args.env):
        key, _, value = item.partition("=")
        monkeypatch.setenv(key, value)
    assert benchmark_vllm._regime_env()["VLLM_ROCM_USE_AITER"] == "0"
    assert benchmark_vllm._regime_env()["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"


def test_explicit_aiter_off_survives_the_default(monkeypatch):
    """``_enable_aiter`` may only supply a default, never overwrite a choice."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    benchmark_vllm._enable_aiter()
    assert benchmark_vllm._regime_env()["VLLM_ROCM_USE_AITER"] == "0"


def test_aiter_defaults_on_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    benchmark_vllm._enable_aiter()
    assert benchmark_vllm._regime_env()["VLLM_ROCM_USE_AITER"] == "1"


# --- two runs under different levers stay distinguishable -------------------

def test_attention_backend_is_part_of_the_regime():
    """An anchor measured on one attention backend cannot serve another."""
    triton = recipe_from_bench_args(bench_args(),
                                    {"VLLM_ROCM_USE_AITER": "1",
                                     "VLLM_ATTENTION_BACKEND": "TRITON_ATTN"})
    default = recipe_from_bench_args(bench_args(),
                                     {"VLLM_ROCM_USE_AITER": "1",
                                      "VLLM_ATTENTION_BACKEND": "ROCM_AITER_FA"})
    assert triton["attention_backend"] == "TRITON_ATTN"
    assert regime_distance(triton, default) == 1
    assert regime_signature(triton) != regime_signature(default)


def test_unset_attention_backend_is_unknown_not_a_match():
    """Absent means "not recorded", which must not be read as a specific backend."""
    recorded = recipe_from_bench_args(bench_args(), {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"})
    unknown = recipe_from_bench_args(bench_args(), {})
    assert unknown["attention_backend"] is None
    # An axis absent on either side is not counted, so an old anchor still
    # matches rather than being spuriously rejected.
    assert regime_distance(recorded, unknown) == 0


def test_aiter_still_separates_regimes():
    on = recipe_from_bench_args(bench_args(), {"VLLM_ROCM_USE_AITER": "1"})
    off = recipe_from_bench_args(bench_args(), {"VLLM_ROCM_USE_AITER": "0"})
    assert regime_distance(on, off) == 1


def test_cache_key_separates_runs_made_under_different_levers(monkeypatch):
    """Two variants must not collide in the result cache and replay each other."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    common = dict(decode_steps=1024, batches="32", bench_layers=None, full_layers=None,
                  benchmark_gpus=1, random_tokens=False, vocab=30000, gpu_mem_util=0.9,
                  routing_dist="none", zipf_s=1.0, moe_imbalance=None, load_format="auto",
                  skip_tokenizer_init=False, max_model_len=None, seed=0, seeds="0,1,2",
                  concurrency=None, decode_context_grid=None)
    a = benchmark_vllm._cache_key(bench_args(server_args="--max-num-seqs 512", env=[], **common))
    b = benchmark_vllm._cache_key(bench_args(server_args="--max-num-seqs 256", env=[], **common))
    c = benchmark_vllm._cache_key(
        bench_args(server_args="--max-num-seqs 512",
                   env=["VLLM_ROCM_USE_AITER_MOE=0"], **common))
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("server_args", ["", "--max-num-seqs 512"])
def test_cache_key_is_stable_for_one_config(monkeypatch, server_args):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    common = dict(decode_steps=1024, batches="32", bench_layers=None, full_layers=None,
                  benchmark_gpus=1, random_tokens=False, vocab=30000, gpu_mem_util=0.9,
                  routing_dist="none", zipf_s=1.0, moe_imbalance=None, load_format="auto",
                  skip_tokenizer_init=False, max_model_len=None, seed=0, seeds="0,1,2",
                  concurrency=None, decode_context_grid=None, server_args=server_args,
                  env=["A=1", "B=2"])
    assert benchmark_vllm._cache_key(bench_args(**common)) == \
        benchmark_vllm._cache_key(bench_args(**common))
