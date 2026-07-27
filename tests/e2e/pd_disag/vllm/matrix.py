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
    # gpu-memory-utilization is capped per-case because RDMA-pinned KV costs TWICE
    # its size in VRAM. Mooncake registers the WHOLE KV reservation, and with no
    # ib_peer_mem on these hosts that goes through ibv_reg_dmabuf_mr, whose import
    # charges another allocation of the same size (measured: exactly 1.000x, once
    # per buffer, released on deregister). So a card only ever fits KV worth half
    # its HBM, and vLLM's own accounting does not see the other half.
    # At 0.4 and tp=2 the KV is ~113 GiB per GPU -> ~226 GiB of a 288 GiB card, and
    # registration starts returning EINVAL; every KV transfer then fails with -1 and
    # the request times out. 0.15 is ~42 GiB per GPU (~85 GiB registered), ample for
    # a 0.6B smoke case. This is a transport constraint, not a tunable: the usual
    # 0.7-0.9 cannot apply when the full KV must be RDMA-pinned.
    [
        QWEN3_0_6B,
        2,
        False,
        False,
        {"server_ready_timeout": 900, "args": ["--gpu-memory-utilization", "0.15"]},
    ],
]


def vllm_disagg_params() -> list:
    """vLLM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
