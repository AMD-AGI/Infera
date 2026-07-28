###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import DEEPSEEK_V4_PRO, GLM_5_1_FP8, GPT_OSS, KIMI_K26_MXFP4, expand_cases

# [model, tp, ep, dp_attn] (+ optional opts dict). A tuple/list on an axis
# enumerates it (e.g. (True, False) runs both). MoE models can exercise ep.
CASES = [
    # gpt-oss-120b: tp2, ep on/off.
    [
        GPT_OSS,
        2,
        (True, False),
        False,
        {"env": {"SGLANG_USE_AITER": "1"}, "server_ready_timeout": 1800},
    ],
    [
        KIMI_K26_MXFP4,
        4,
        True,
        True,
        {
            "env": {"SGLANG_USE_AITER": "1"},
            # Multithreaded weight load (forwarded verbatim to sglang's
            # ServerArgs / launch_server) to speed up loading the many shards.
            "args": [
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            "server_ready_timeout": 1800,
        },
    ],
    # DeepSeek-V4-Pro (MoE, tp8): --attention-backend dsv4 selects the DSv4 sparse
    # attention; the SGLANG_OPT_*/AITER env is its FP8 config. tp from the adapter.
    [
        DEEPSEEK_V4_PRO,
        8,
        False,
        False,
        {
            "args": [
                "--attention-backend",
                "dsv4",
                "--page-size",
                "256",
                "--disable-radix-cache",
                "--disable-shared-experts-fusion",
                "--swa-full-tokens-ratio",
                "0.15",
                "--mem-fraction-static",
                "0.90",
                "--chunked-prefill-size",
                "8192",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "SGLANG_OPT_FP8_WO_A_GEMM": "0",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
                "SGLANG_OPT_USE_AITER_INDEXER": "1",
                "SGLANG_OPT_USE_TOPK_V2": "0",
                "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
                "SGLANG_OPT_USE_FUSED_PAGED_COMPRESS": "1",
                "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
                "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
                "SGLANG_ROCM_USE_MULTI_STREAM": "false",
                "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
                "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
                "SGLANG_EAGER_INPUT_NO_COPY": "true",
                "SGLANG_USE_ROCM700A": "0",
                "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
            },
            "server_ready_timeout": 2400,
        },
    ],
    # GLM-5.1-FP8 (GlmMoeDsa = MLA + DSA lightning indexer, tp4). Minimal ON PURPOSE:
    # SGLang routes GlmMoeDsaForCausalLM through the DeepSeek MLA+DSA path and
    # auto-selects attention_backend=dsa / page_size=64 / tilelang / kv bf16 — do NOT
    # force the DSv4 flags (--attention-backend dsv4, --page-size 256), they fight the
    # auto-config. --reasoning-parser glm45 splits GLM reasoning_content; AITER on.
    # Verified 2026-07-23 single-node mix, temp=0: France->Paris/China->Beijing/2+2->4.
    # Long timeout covers the ~8-10 min silent tilelang-JIT + aiter-GEMM-tuning window.
    [
        GLM_5_1_FP8,
        4,
        False,
        False,
        {
            "args": [
                "--reasoning-parser",
                "glm45",
                "--mem-fraction-static",
                "0.85",
            ],
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 1800,
        },
    ],
]


def sglang_mixed_params() -> list:
    """SGLang matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
