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

from ...harness.matrix import GPT_OSS, expand_cases

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
]


def vllm_disagg_params() -> list:
    """vLLM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
