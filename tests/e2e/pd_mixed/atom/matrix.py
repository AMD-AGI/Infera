###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import DEEPSEEK_V4_PRO, GLM_5_1_FP8, GPT_OSS, KIMI_K26_MXFP4, expand_cases

# [enable, model, tp, ep, dp_attn] (+ optional opts dict). Opts mirror the
# matching InferenceX single_node/fixed_seq_len benchmarks.
CASES = [
    [
        True,
        GPT_OSS,
        2,
        True,
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
            # ATOM intends to support this: it routes gfx94x MXFP4 MoE to its
            # aiter-triton kernels on purpose (moe.py picks use_triton for
            # gfx94x, and gfx942 has its own e4m3fnuz/CDNA4-swizzle branches).
            # No published image makes that path run, though -- see
            # manual/wip/gfx942-atom-gpt-oss.md for the three tags measured.
            "gfx942": {"skip": "ATOM's gfx942 MXFP4 MoE path is broken upstream"},
        },
    ],
    [
        True,
        KIMI_K26_MXFP4,
        4,
        False,
        False,
        {
            "args": ["--kv_cache_dtype", "fp8", "--trust-remote-code"],
            "env": {"OMP_NUM_THREADS": "1"},
            "server_ready_timeout": 1800,
            # Another MXFP4 checkpoint, so it likely takes the same gfx942 MoE
            # route that is broken upstream for gpt-oss above — but nobody has
            # run it, and ATOM has no runtime knob to steer it either way.
            "gfx942": {"skip": "Kimi-K2.6 MXFP4 not measured on gfx942 yet"},
        },
    ],
    [
        False,
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
    # GLM-5.1-FP8 (GlmMoeDsa = MLA + DSA lightning indexer, tp4). ATOM loads
    # GlmMoeDsaForCausalLM natively (allocates the MLA chunked-prefill workspaces).
    # Minimal ON PURPOSE: fp8 KV is enough; NO --method mtp (GLM ships no MTP/nextn
    # draft weights, and gfx950 plain decode is correct — the gfx942 broken-plain-
    # decode bug that forced MTP on DSv4 does not reproduce). GLM ships chat_template
    # so /v1/chat/completions works. Verified 2026-07-23 single-node, temp=0:
    # France->Paris/China->Beijing/2+2->4 (thinking disabled in the probe).
    [
        True,
        GLM_5_1_FP8,
        4,
        False,
        False,
        {
            "args": ["--kv_cache_dtype", "fp8", "--trust-remote-code"],
            "env": {"OMP_NUM_THREADS": "1", "HSA_NO_SCRATCH_RECLAIM": "1"},
            "server_ready_timeout": 1800,
            # Not measured on gfx942 in this suite: fp8 sidesteps the MXFP4
            # breakage that skips gpt-oss above, but this tp4 row has never
            # been fitted against MI300X's 192 GB per card (MI355X has 288).
            "gfx942": {"skip": "GLM-5.1-FP8 not measured on gfx942 yet"},
        },
    ],
]


def atom_mixed_params() -> list:
    """ATOM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
