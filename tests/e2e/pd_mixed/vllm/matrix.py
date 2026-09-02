###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import (
    DEEPSEEK_V4_PRO,
    GLM_5_1_FP8,
    GPT_OSS,
    KIMI_K26_MXFP4,
    QWEN3_8B,
    expand_cases,
)

# [enable, model, tp, ep, dp_attn] (+ optional opts dict). Opts mirror the
# matching InferenceX single_node/fixed_seq_len benchmarks.
CASES = [
    # gpt-oss-120b, tp2. The other three engines' PD-mixed grids all carry gpt-oss but
    # this one did not, which left vLLM as the only engine covered in PD-disag and not
    # in PD-mixed. Config is deliberately the same as the pd_disag/vllm row for the
    # same model, so a difference between the two scenarios is a scenario difference
    # and not a knob difference. tp2 also keeps it schedulable on this cluster, where
    # the tp4 rows below need four idle cards on one node.
    #
    # --use-fp64-gumbel is a ROCm 10.1 workaround, not a tuning choice. aiter's
    # top-k/top-p sampling kernel does not compile under ROCm 10.1 — its
    # sampling.cuh still uses hipcub::Traits<T>, which hipCUB has dropped, so the
    # JIT build fails with "no member named 'Traits' in namespace 'hipcub'". vLLM
    # reaches that kernel from its memory-profiling dummy run, so the server never
    # reports ready. Turning aiter off wholesale is not an option here: gpt-oss is
    # MXFP4 and vLLM has no native MXFP4 MoE backend (which is why infera defaults
    # VLLM_ROCM_USE_AITER=1). This flag sends only the sampler down the
    # PyTorch-native path and leaves aiter serving the MoE. The cases below assert
    # on greedy decoding, which the Gumbel-noise dtype does not affect. Drop it once
    # aiter's sampling.cuh builds against ROCm 10.1's hipCUB.
    [
        True,
        GPT_OSS,
        2,
        False,
        False,
        {
            "server_ready_timeout": 1800,
            "args": ["--gpu-memory-utilization", "0.9", "--use-fp64-gumbel"],
        },
    ],
    [
        False,
        QWEN3_8B,
        2,
        False,
        False,
        {
            "args": ["--kv-cache-dtype", "fp8"],
            "env": {"VLLM_ROCM_USE_AITER": "1"},
        },
    ],
    [
        True,
        KIMI_K26_MXFP4,
        4,
        False,
        False,
        {
            "args": [
                "--gpu-memory-utilization",
                "0.90",
                "--max-model-len",
                "10240",
                "--block-size",
                "1",
                "--mm-encoder-tp-mode",
                "data",
            ],
            "env": {
                "VLLM_ROCM_USE_AITER": "1",
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
                "VLLM_ROCM_USE_AITER_RMSNORM": "0",
                "HSA_NO_SCRATCH_RECLAIM": "1",
            },
            # MXFP4 needs amd-quark; the vllm image already ships it, this is a
            # safety net mirroring the InferenceX recipe (no-op if present).
            "setup": ["pip install amd-quark"],
            "server_ready_timeout": 1800,
        },
    ],
    # DeepSeek-V4-Pro (MoE, tp8): DSv4 needs the deepseek_v4 tokenizer + reasoning
    # parser; aiter MoE backend + fp8 KV are its config. tp from the adapter.
    [
        False,
        DEEPSEEK_V4_PRO,
        8,
        False,
        False,
        {
            "args": [
                "--kv-cache-dtype",
                "fp8",
                "--moe-backend",
                "aiter",
                "--tokenizer-mode",
                "deepseek_v4",
                "--reasoning-parser",
                "deepseek_v4",
                "--no-enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.90",
                "--max-model-len",
                "9472",
                "--max-num-batched-tokens",
                "8192",
                "--max-num-seqs",
                "128",
                "--distributed-executor-backend",
                "mp",
                "--disable-hybrid-kv-cache-manager",
                "--async-scheduling",
                "--compilation-config",
                '{"max_cudagraph_capture_size":128}',
            ],
            "env": {
                "VLLM_USE_V1": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
                "PYTHONHASHSEED": "0",
            },
            "server_ready_timeout": 2400,
        },
    ],
    # GLM-5.1-FP8 (GlmMoeDsa = MLA + DSA lightning indexer, tp4). vLLM v0.25.1 serves
    # it via the DeepSeek MLA path; fp8 KV + aiter (via env) + the glm45 reasoning
    # parser is its config. Single-node mix uses NO kv-transfer connector (that's the
    # pd_disag/vllm MoRIIO/Mooncake path). Verified 2026-07-23 single-node mix, temp=0
    # (thinking disabled): France->Paris/China->Beijing/2+2->4.
    [
        True,
        GLM_5_1_FP8,
        4,
        False,
        False,
        {
            "args": [
                "--kv-cache-dtype",
                "fp8",
                "--reasoning-parser",
                "glm45",
                "--no-enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.85",
                "--max-model-len",
                "9472",
                "--max-num-batched-tokens",
                "8192",
                "--distributed-executor-backend",
                "mp",
            ],
            "env": {
                "VLLM_USE_V1": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
                "PYTHONHASHSEED": "0",
            },
            "server_ready_timeout": 1800,
        },
    ],
]


def vllm_mixed_params() -> list:
    """vLLM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
