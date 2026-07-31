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
            "args": ["--mem-fraction-static", "0.9"],
        },
    ],
]


def sglang_disagg_params() -> list:
    """SGLang PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
