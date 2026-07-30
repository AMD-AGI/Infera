###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import KIMI_K26_MXFP4, QWEN3_8B, expand_cases

# [model, tp, ep, dp_attn] (+ optional opts dict). Opts mirror the matching
# InferenceX single_node/fixed_seq_len benchmarks.
CASES = [
    [
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
]


def vllm_mixed_params() -> list:
    """vLLM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
