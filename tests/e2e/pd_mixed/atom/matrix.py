###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import DEEPSEEK_V4_PRO, GPT_OSS, KIMI_K26_MXFP4, expand_cases

# [model, tp, ep, dp_attn] (+ optional opts dict). Opts mirror the matching
# InferenceX single_node/fixed_seq_len benchmarks.
CASES = [
    [
        GPT_OSS,
        2,
        False,
        False,
        {
            "args": [
                "--kv_cache_dtype",
                "fp8",
                "--gpu-memory-utilization",
                "0.9",
                "--block-size",
                "16",
            ],
            "env": {"ATOM_GPT_OSS_MODEL": "1", "OMP_NUM_THREADS": "1"},
            "server_ready_timeout": 1800,
        },
    ],
    [
        KIMI_K26_MXFP4,
        4,
        (True, False),
        False,
        {
            "args": ["--kv_cache_dtype", "fp8", "--trust-remote-code"],
            "env": {"OMP_NUM_THREADS": "1"},
            "server_ready_timeout": 1800,
        },
    ],
    [
        DEEPSEEK_V4_PRO,
        8,
        False,
        False,
        {
            "args": [
                "--kv_cache_dtype",
                "fp8",
                "--max-model-len",
                "16384",
                "--gpu-memory-utilization",
                "0.9",
                "--cudagraph-capture-sizes",
                "[1,2,4,8]",
                "--hf-overrides",
                '{"use_index_cache": true, "index_topk_freq": 4}',
                "--trust-remote-code",
            ],
            "env": {
                "OMP_NUM_THREADS": "1",
                "ATOM_DISABLE_MMAP": "true",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "ATOM_MOE_GU_ITLV": "1",
                "INFERA_ATOM_READY_TIMEOUT": "2700",
            },
            # ~25min weight load (mmap off) + cudagraph capture; INFERA_ATOM_READY_TIMEOUT
            # is set to 2700s above (default is 1800s) so the worker doesn't time out early.
            "server_ready_timeout": 2700,
        },
    ],
]


def atom_mixed_params() -> list:
    """ATOM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
