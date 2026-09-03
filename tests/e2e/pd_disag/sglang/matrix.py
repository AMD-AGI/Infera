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

from ...harness.matrix import (
    DEEPSEEK_V4_FLASH_FP8,
    GLM_5_2_FP8,
    GPT_OSS,
    expand_cases,
)

# See the PD-mixed table's copy of this: brought up on the local MI300X fleet,
# never run on CI, and the skip claims only what is known.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

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
            # Follow SGLang's native platform gate: gfx942 is not a supported
            # MXFP4 target for this checkpoint, so do not force a fallback.
            "gfx942": {
                "skip": "SGLang's upstream MXFP4 gate excludes gpt-oss-120b on gfx942",
            },
        },
    ],
    # GLM-5.2-FP8, prefill TP8/DP8 + decode TP8/DP8, KV over Mooncake RDMA — the
    # `disaggregated` arm of manual/recipes/glm5.2-fp8-gfx942.md. Both legs get
    # identical argv from this one row, which is what SGLang requires: it rejects
    # a PD pair whose speculative config or attention parallelism disagrees.
    #
    # This is the first row here to need a whole node per leg, so the tier's two
    # held nodes are now fully committed rather than using 2 of 8 GPUs each.
    #
    # No --disaggregation-ib-device: the rail is a property of the cluster, not of
    # the case, and the site profile pins it for every engine at once through
    # MC_TE_FILTERS (tests/sites/mi300x-rccl.env).
    [
        True,
        GLM_5_2_FP8,
        8,
        False,
        True,
        {
            "args": [
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--mem-fraction-static",
                "0.85",
                "--max-running-requests",
                "128",
                "--chunked-prefill-size",
                "8192",
                "--watchdog-timeout",
                "1200",
                "--disable-custom-all-reduce",
                "--enable-cache-report",
                # DP attention round-robins probes across eight schedulers.
                # Export all of them so MTP activity is not hidden behind DP0.
                "--enable-metrics-for-all-schedulers",
                # E2E probes are too short for SGLang's 40-step default flush.
                "--decode-log-interval",
                "1",
                "--weight-loader-prefetch-checkpoints",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 32}',
                # MTP survives disaggregation, and is in fact safer here than on a
                # mixed worker: verification happens only on the decode leg, so the
                # MTP-plus-hicache hang the recipe documents cannot form. The
                # prefill leg reports spec_accept_length 0.0 by design.
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "6",
                "--json-model-override-args",
                '{"index_share_for_mtp_iteration":false}',
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_DSA_TRITON_PREFILL": "1",
                "SAFETENSORS_FAST_GPU": "1",
                "HSA_NO_SCRATCH_RECLAIM": "1",
                "INFERA_ENGINE_READY_TIMEOUT": "5400",
                # Registration comes after the weights load, and the default
                # gives up long before two legs of this size are both up.
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
            },
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
    # Flash-FP8 uses TP4 on each leg. Keep MTP flags explicit so the harness can
    # verify drafting on the decode worker.
    [
        True,
        DEEPSEEK_V4_FLASH_FP8,
        4,
        False,
        True,
        {
            "args": [
                "--attention-backend",
                "dsv4",
                "--page-size",
                "256",
                "--disable-radix-cache",
                "--disable-shared-experts-fusion",
                "--mem-fraction-static",
                "0.80",
                "--chunked-prefill-size",
                "8192",
                "--context-length",
                "9472",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 32}',
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--enable-metrics-for-all-schedulers",
                "--decode-log-interval",
                "1",
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "SGLANG_OPT_FP8_WO_A_GEMM": "0",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
                "SGLANG_OPT_USE_AITER_INDEXER": "1",
                "SGLANG_OPT_USE_TOPK_V2": "0",
                "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
                "SGLANG_OPT_USE_FUSED_PAGED_COMPRESS": "1",
                "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
                "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
                "SGLANG_ROCM_USE_MULTI_STREAM": "false",
                "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
                "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
                "SGLANG_EAGER_INPUT_NO_COPY": "true",
                "SGLANG_USE_ROCM700A": "0",
                "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
                "HSA_NO_SCRATCH_RECLAIM": "1",
                # Registration comes after the weights load, and the default
                # gives up long before both legs are up.
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
                "INFERA_ENGINE_READY_TIMEOUT": "3600",
            },
            "server_ready_timeout": 3600,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
]


def sglang_disagg_params() -> list:
    """SGLang PD-disaggregated matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
