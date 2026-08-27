###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang PD-disaggregated parametrize grid. Declarative ``CASES`` table — same
row/axis semantics as the PD-mixed grids (see harness/matrix.py).

Add a case = add ONE row. Each row spawns a cross-node prefill+decode pair.
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
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 1800,
            # triton, not the default aiter backend: the CK batch_prefill instance
            # this case needs (page_size < kN0 over a >2GB KV cache, gfx950) is
            # absent from the aiter in the v0.5.17 base, so both TP ranks raise
            # "no matching kernel found" and the prefill leg dies. Drop this once a
            # base image carries the instance; aiter stays on for MoE either way.
            "args": ["--mem-fraction-static", "0.9", "--attention-backend", "triton"],
            # gfx942 has no FP4 MFMA, so aiter's CK-tile MXFP4 MoE carries no
            # instance for it and graph capture dies on an undefined symbol.
            # Off, the MoE falls to Triton, which dequantizes to bf16. The
            # runner is pinned because at ep_size 1 sglang forces gpt-oss onto
            # triton_kernel, and this image's triton_kernels has no matmul_ogs.
            "gfx942": {
                "env": {"SGLANG_USE_AITER": "0"},
                "args": [
                    "--mem-fraction-static",
                    "0.9",
                    "--attention-backend",
                    "triton",
                    "--moe-runner-backend",
                    "triton",
                ],
            },
        },
    ],
]


def sglang_disagg_params() -> list:
    """SGLang PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
