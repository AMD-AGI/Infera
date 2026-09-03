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
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V4_PRO,
    GLM_5_1_FP8,
    GLM_5_2_FP8,
    GPT_OSS,
    KIMI_K26_MXFP4,
    QWEN3_8B,
    expand_cases,
)

# The gfx942 fleet is where GLM-5.2 is staged and being brought up; whether the
# gfx950 CI fleet holds it at all is unconfirmed, so the skip claims only what is
# known rather than asserting a staging fact.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

# [enable, model, tp, ep, dp_attn] (+ optional opts dict). Opts mirror the
# matching InferenceX single_node/fixed_seq_len benchmarks.
CASES = [
    # pd_disag/vllm's gpt-oss knobs on one node, plus ep as the other mixed grids do:
    # single-node serving is the precondition for reading a cross-node KV failure.
    [
        True,
        GPT_OSS,
        2,
        True,
        False,
        {
            "args": ["--gpu-memory-utilization", "0.9"],
            "server_ready_timeout": 1800,
            # aiter's MXFP4 MoE returns garbage logits on gfx942 (a wall of "!");
            # turning it off falls back to the Triton MXFP4 path, which is correct.
            "gfx942": {"env": {"VLLM_ROCM_USE_AITER": "0"}},
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
            # MXFP4 on aiter, the pairing that returned garbage logits for
            # gpt-oss above; whether the same VLLM_ROCM_USE_AITER=0 fallback
            # serves this row is one run away. Skip until someone takes it —
            # gfx950's knobs would only fail in a way nobody has read.
            "gfx942": {"skip": "Kimi-K2.6 MXFP4 not measured on gfx942 yet"},
        },
    ],
    # DeepSeek-V4-Pro (MoE, tp8): DSv4 needs the deepseek_v4 tokenizer + reasoning
    # parser; aiter MoE backend + fp8 KV are its verified MI355X config. tp from
    # the adapter. Keep architecture-specific changes in an overlay below.
    [
        True,
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
            # gfx942 drops --moe-backend: Pro's routed experts are MXFP4, and for
            # MXFP4 layers `aiter` selects the AITER_MXFP4_* family — the kernels
            # the gpt-oss row above disables here for returning garbage logits.
            # Unset, vLLM's per-quantization oracle should pick TRITON_UNFUSED,
            # which upcasts fp4->bf16 in-kernel so the experts stay packed rather
            # than double.
            "gfx942": {
                "args": [
                    "--kv-cache-dtype",
                    "fp8",
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
                    # The checkpoint ships one draft layer, stepped 3 times —
                    # the depth its PD-disagg twin and both Flash rows use.
                    "--speculative-config",
                    '{"method":"mtp","num_speculative_tokens":3}',
                ],
                "server_ready_timeout": 5400,
            },
            "gfx950": {"skip": "manual-only: 806 GiB checkpoint is too costly for routine PR CI"},
        },
    ],
    # DeepSeek-V4-Flash (MoE, tp4) — Pro's architecture one size down (43 layers /
    # 4096 hidden against 61 / 7168) and the same hybrid quantization. This is a
    # new gfx942 recipe, not a modification of the verified Pro base above.
    #
    # It earns its own row rather than replacing Pro's because it is the size that
    # makes dsv4 routine here: 149 GiB against 806, which is 37.2 a card at tp4
    # and a load measured in minutes rather than the hour and forty Pro takes.
    # Pro remains manual-only on gfx950 because its 806 GiB load is too costly for
    # routine PR CI; both rows run in this gfx942 bring-up.
    #
    # tp4 for the same reason as its SGLang twin. One caveat if it misbehaves
    # before anything else does: `o_groups: 8` gives 2 output groups a rank here
    # against 1 at tp8, so a shape assertion in the attention output projection is
    # the first place to look, and tp8 is the cheap thing to try.
    [
        True,
        DEEPSEEK_V4_FLASH,
        4,
        False,
        False,
        {
            "args": [
                "--kv-cache-dtype",
                "fp8",
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
                # The checkpoint ships one draft layer, same as Pro.
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":3}',
            ],
            "env": {
                "VLLM_USE_V1": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
                "PYTHONHASHSEED": "0",
            },
            "server_ready_timeout": 2400,
            "gfx950": _GFX950_UNMEASURED,
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
            # fp8, so gpt-oss's MXFP4 findings say nothing here; and this tp4
            # row has never been fitted against MI300X's 192 GB per card
            # (MI355X has 288). Both are one run to settle.
            "gfx942": {"skip": "GLM-5.1-FP8 not measured on gfx942 yet"},
        },
    ],
    # GLM-5.2-FP8 (GlmMoeDsa, tp8). A DIFFERENT model from the GLM-5.1 row above
    # (78 layers, index_topk 2048), not a newer tag for it, so both stand. vLLM
    # serves the architecture through the same DeepSeek MLA path GLM-5.1 uses;
    # what is new here is the size — ~700 GB of FP8 weights needs all eight
    # cards, where GLM-5.1 fits in four.
    #
    # MTP because the checkpoint ships the draft head (num_nextn_predict_layers
    # 1). vLLM reaches it by a different route than SGLang: config/speculative.py
    # rewrites model_type glm_moe_dsa -> deepseek_mtp -> DeepSeekMTPModel, and
    # `method: mtp` is the spelling that survives — "deepseek_mtp" is accepted
    # but deprecated, and only "mtp" defaults the draft `model` to the target's
    # path (the draft weights live in the target checkpoint).
    [
        True,
        GLM_5_2_FP8,
        8,
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
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":3}',
            ],
            "env": {
                "VLLM_USE_V1": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
                "PYTHONHASHSEED": "0",
            },
            # ~700 GB over NFS, then the draft head and graph capture on top.
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
]


def vllm_mixed_params() -> list:
    """vLLM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
