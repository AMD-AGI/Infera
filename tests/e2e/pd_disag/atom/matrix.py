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

import json

import pytest

from ...harness.matrix import (
    GLM_5_2_FP8,
    GLM_5_2_INDEXER_PATTERN,
    GPT_OSS,
    expand_cases,
)

# The gfx942 fleet is where GLM-5.2 is staged and being brought up; whether the
# gfx950 CI fleet holds it at all is unconfirmed, so the skip claims only what is
# known rather than asserting a staging fact.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

# [enable, model, tp, ep, dp_attn] (+ optional opts: args/env/setup/server_ready_timeout).
CASES = [
    # Debug: bigger model (gpt-oss-120b), prefill TP=2 + decode TP=2, Mooncake RDMA.
    [
        True,
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
            # Same upstream MXFP4 MoE gap as the PD-mixed row; the model cannot
            # load, so KV transfer never gets a chance to matter. See
            # tests/e2e/pd_mixed/atom/matrix.py.
            "gfx942": {"skip": "ATOM's gfx942 MXFP4 MoE path is broken upstream"},
        },
    ],
    # GLM-5.2-FP8, prefill TP8 + decode TP8. Same knobs as the PD-mixed ATOM row
    # (tests/e2e/pd_mixed/atom/matrix.py); the adapter supplies what is
    # disaggregation-specific. FP8, so this row does not reach the MXFP4 MoE path
    # that skips gpt-oss above — it is the only ATOM case expected to run on
    # gfx942, and it only gets here once its PD-mixed twin is green.
    [
        True,
        GLM_5_2_FP8,
        8,
        False,
        False,
        {
            "args": [
                "--kv_cache_dtype",
                "fp8",
                "--trust-remote-code",
                "--max-model-len",
                "9472",
                "--gpu-memory-utilization",
                "0.85",
                # GLM-5.2's indexer sharing, same as the PD-mixed row — without
                # it 57 of 78 layers run an indexer the checkpoint never filled
                # in. See tests/e2e/pd_mixed/atom/matrix.py for the mechanism.
                "--hf-overrides",
                json.dumps(
                    {
                        "use_index_cache": True,
                        "index_topk_pattern": GLM_5_2_INDEXER_PATTERN,
                    }
                ),
                # NO --method mtp, unlike this row's PD-mixed twin, which passes
                # with it (38m, all three probes). With MTP here both workers come
                # up and register and the correctness probe then never returns —
                # httpx.ReadTimeout after 49m, a hang rather than a wrong answer.
                # Without it the row passes in 34m, so ATOM's PD path itself is
                # sound and speculation across the boundary is what is not.
                #
                # Third engine, same axis: vLLM's PD row for this model answers
                # wrongly with MTP on and correctly without, and SGLang's needs
                # index_share_for_mtp_iteration false or it hangs. That override
                # is not available here — the key is inert in ATOM's tree, which
                # has no reference to it — so this row has no knob to try.
            ],
            "env": {
                "OMP_NUM_THREADS": "1",
                "HSA_NO_SCRATCH_RECLAIM": "1",
                "INFERA_ATOM_READY_TIMEOUT": "5400",
            },
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
]


def atom_disagg_params() -> list:
    """ATOM PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
