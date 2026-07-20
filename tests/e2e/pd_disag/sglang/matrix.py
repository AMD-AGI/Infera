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

# [model, tp, ep, dp_attn] (+ optional opts dict: args/env/setup/server_ready_timeout).
CASES = [
    # Debug: bigger model (gpt-oss-120b). Larger weights leave a smaller KV pool,
    # so Mooncake's RDMA registration may stay under ionic's 2 GiB ibv_reg_mr cap
    # that the small-model case tripped.
    [
        GPT_OSS,
        2,
        False,
        False,
        {
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 1800,
            "args": ["--mem-fraction-static", "0.5"],
        },
    ],
]


def sglang_disagg_params() -> list:
    """SGLang PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
