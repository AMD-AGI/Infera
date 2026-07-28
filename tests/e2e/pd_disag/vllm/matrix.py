###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM PD-disaggregated parametrize grid. Declarative ``CASES`` table — same
row/axis semantics as the PD-mixed grids (see harness/matrix.py).

Add a case = add ONE row. Each row spawns a cross-node prefill+decode pair for
that model/knobs. Keep the default case tiny (Qwen3-0.6B, tp1) so a PD smoke run
is fast; larger models go in their own rows with per-case ``opts``.
"""

from __future__ import annotations

import pytest

from ...harness.matrix import GLM_5_1_FP8, QWEN3_0_6B, expand_cases

# [model, tp, ep, dp_attn] (+ optional opts dict: args/env/setup/server_ready_timeout).
CASES = [
    # Small dense smoke case: 1 prefill GPU + 1 decode GPU, Mooncake RDMA.
    # server_ready_timeout is generous for cross-node cold start + bootstrap.
    # gpu-memory-utilization is capped per-case: Mooncake pins/registers the WHOLE
    # KV reservation for RDMA, so it's bounded by (a) ionic's ibv_reg_mr ceiling
    # (tp=1: 0.5 ~142 GiB OK, >=0.6 segfaults) and (b) at tp>1 an extra registered
    # buffer per GPU (tp=2 @0.5 OOMs; 0.4 fits). This is a transport constraint, not
    # a tunable — the usual 0.7-0.9 can't apply when the full KV must be RDMA-pinned.
    [
        QWEN3_0_6B,
        2,
        False,
        False,
        {"server_ready_timeout": 900, "args": ["--gpu-memory-utilization", "0.4"]},
    ],
    # GLM-5.1-FP8 over the MoRIIO connector (NOT Mooncake) — the regression guard for
    # the moriio_layout.py MLA page-length fix (patch_moriio_pagelen.py). MoRIIO is
    # opted into explicitly via env[INFERA_E2E_KV_CONNECTOR]; without it the disagg
    # adapter defaults to Mooncake, so this is the only MoRIIO case and no other model
    # gets a MoRIIO variant. GlmMoeDsa = block-scaled fp8 MLA + DSA lightning indexer,
    # the exact layout the page-len bug corrupted. args mirror the verified launch
    # recipe (fp8 KV, aiter MoE, glm45 reasoning parser); timeout covers CG capture.
    [
        GLM_5_1_FP8,
        4,
        False,
        False,
        {
            "env": {"INFERA_E2E_KV_CONNECTOR": "MoRIIOConnector"},
            "args": [
                "--kv-cache-dtype",
                "fp8",
                "--moe-backend",
                "aiter",
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
            "server_ready_timeout": 1800,
        },
    ],
]


def vllm_disagg_params() -> list:
    """vLLM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
