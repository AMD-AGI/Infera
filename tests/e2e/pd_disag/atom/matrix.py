###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM PD-disaggregated parametrize grid. Declarative ``CASES`` table — same
row/axis semantics as the PD-mixed grids (see harness/matrix.py).

Add a case = add ONE row. Each row spawns a cross-node prefill+decode pair.
"""

from __future__ import annotations

import pytest

from ...harness.matrix import GPT_OSS, expand_cases

# [model, tp, ep, dp_attn] (+ optional opts dict: args/env/setup/server_ready_timeout).
CASES = [
    # Debug: bigger model (gpt-oss-120b), prefill TP=2 + decode TP=2, Mooncake RDMA.
    # gpu-memory-utilization is bounded by the RDMA registration, not by the KV pool
    # alone: RDMA-pinned KV costs twice its size (ibv_reg_dmabuf_mr charges a second
    # allocation of the same size — see the vLLM grid for the measurement), and ATOM's
    # budget line does not account for that half. 0.9 leaves available_for_kv=215 GiB
    # PER GPU, i.e. ~430 GiB wanted on a 288 GiB card, so the model load itself dies
    # with 'HIP out of memory'. 0.4 lands near 70 GiB of KV per GPU (~140 GiB with the
    # registration), which fits with headroom.
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
                "0.4",
                "--block-size",
                "16",
            ],
            "env": {"ATOM_GPT_OSS_MODEL": "1", "OMP_NUM_THREADS": "1"},
            "server_ready_timeout": 1800,
        },
    ],
]


def atom_disagg_params() -> list:
    """ATOM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
