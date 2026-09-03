###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import json

import pytest

from ...harness.matrix import (
    DEEPSEEK_V4_PRO,
    GLM_5_1_FP8,
    GLM_5_2_FP8,
    GLM_5_2_INDEXER_PATTERN,
    GPT_OSS,
    KIMI_K26_MXFP4,
    expand_cases,
)

# The gfx942 fleet is where GLM-5.2 is staged and being brought up; whether the
# gfx950 CI fleet holds it at all is unconfirmed, so the skip claims only what is
# known rather than asserting a staging fact.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

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
            # No published image makes that path run.
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
            # Two measured facts, and the second is why the first was not worth
            # fixing. ATOM has no working MXFP4 expert kernel on gfx942 — all
            # three of its paths fail, the last one because aiter's CK MX-FP4
            # kernel has no gfx942 device image and CDNA3 has no FP4 MFMA to
            # build one — so the checkpoint would need an engine-side FP4->FP8
            # dequant to load at all. Unpacked it then needs 195.8 GiB a card at
            # tp8 against 191.98 usable, which is where SGLang's copy of this
            # exercise ended. So the dsv4 row ATOM serves here is
            # DeepSeek-V4-Flash-FP8 below.
            "gfx942": {
                "skip": "no gfx942 MXFP4 expert kernel (CDNA3 has no FP4 MFMA), and dequanting to FP8 needs 195.8 GiB/card against 191.98; run DeepSeek-V4-Flash-FP8 here, or Pro on vLLM which serves it packed",
            },
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
    # GLM-5.2-FP8 (GlmMoeDsa, tp8) — a different model from the GLM-5.1 row above,
    # not a newer tag for it: 78 layers, index_topk 2048, and ~700 GB of FP8 weights
    # that need all eight cards. Same deepseek_v2 load path.
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
                # ATOM builds an Indexer on every layer (deepseek_v2.py:1809), but
                # GLM-5.2 ships weights only for the 21 "full" layers, so without this
                # 57 run randomly-initialised ones — 399 of 1947 params "NOT loaded".
                #
                # Spelled out rather than ATOM's index_topk_freq shorthand: that formula
                # picks 0,1,5,9... where GLM-5.2 owns 0,1,2,6,10... — the
                # index_skip_topk_offset 3 that ATOM does not read.
                #
                # Length 78 leaves the MTP layer (78) out of range, so it computes its
                # own, which is correct: the checkpoint ships indexer weights for it.
                "--hf-overrides",
                json.dumps(
                    {
                        "use_index_cache": True,
                        "index_topk_pattern": GLM_5_2_INDEXER_PATTERN,
                    }
                ),
                "--method",
                "mtp",
                "--num-speculative-tokens",
                "3",
            ],
            "env": {
                "OMP_NUM_THREADS": "1",
                "HSA_NO_SCRATCH_RECLAIM": "1",
                # ATOM's own patience, separate from the harness's below; the
                # shorter of the two wins, so both cover the same cold start.
                "INFERA_ATOM_READY_TIMEOUT": "5400",
            },
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
]


def atom_mixed_params() -> list:
    """ATOM matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
