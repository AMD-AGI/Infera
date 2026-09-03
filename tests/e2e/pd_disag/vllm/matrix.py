###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM PD-disaggregated parametrize grid. Declarative ``CASES`` table — same
row/axis semantics as the PD-mixed grids (see harness/matrix.py).

Add a case = add ONE row. Each row spawns a cross-node prefill+decode pair for
that model/knobs.
"""

from __future__ import annotations

import pytest

from ...harness.matrix import (
    DEEPSEEK_V4_FLASH,
    DEEPSEEK_V4_PRO,
    GLM_5_2_FP8,
    GPT_OSS,
    expand_cases,
)

# The gfx942 fleet is where GLM-5.2 is staged and being brought up; whether the
# gfx950 CI fleet holds it at all is unconfirmed, so the skip claims only what is
# known rather than asserting a staging fact.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

# [enable, model, tp, ep, dp_attn] (+ optional opts: args/env/setup/server_ready_timeout).
CASES = [
    # gpt-oss-120b, prefill TP=2 + decode TP=2, KV over Mooncake RDMA.
    [
        True,
        GPT_OSS,
        2,
        False,
        False,
        {
            "server_ready_timeout": 1800,
            "args": ["--gpu-memory-utilization", "0.9"],
            # aiter's MXFP4 MoE returns garbage logits on gfx942 (a wall of "!");
            # off, vLLM falls back to the Triton MXFP4 path, which is correct.
            "gfx942": {"env": {"VLLM_ROCM_USE_AITER": "0"}},
        },
    ],
    # GLM-5.2-FP8, prefill TP8 + decode TP8. Same knobs as the PD-mixed vLLM row
    # (tests/e2e/pd_mixed/vllm/matrix.py) — the adapter supplies everything that
    # is disaggregation-specific (kv-transfer-config, advertise-host, bootstrap
    # port), so what differs here is only that each leg now takes a whole node.
    #
    # Keep MTP off: MTP3 passes short prompts but corrupts the long-context
    # result, including with disable_padded_drafter_batch enabled.
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
            ],
            "env": {
                "VLLM_USE_V1": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
                "PYTHONHASHSEED": "0",
            },
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
    # Pro uses packed MXFP4 experts at TP8. The PD row omits mixed-only async
    # scheduling and graph-capture sizing; the adapter supplies --max-num-seqs.
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
                "--distributed-executor-backend",
                "mp",
                "--disable-hybrid-kv-cache-manager",
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
            "server_ready_timeout": 5400,
            "gfx950": {"skip": "manual-only: 806 GiB checkpoint is too costly for routine PR CI"},
        },
    ],
    # Flash uses the same packed-MXFP4 path at TP4. MTP remains explicit so the
    # harness can verify that drafting is active on the decode leg.
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
                "--distributed-executor-backend",
                "mp",
                "--disable-hybrid-kv-cache-manager",
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
]


def vllm_disagg_params() -> list:
    """vLLM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
