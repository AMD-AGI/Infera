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

from ...harness.matrix import QWEN3_0_6B, expand_cases

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
]


def vllm_disagg_params() -> list:
    """vLLM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
