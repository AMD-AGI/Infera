###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang e2e parametrize grid. Declarative ``CASES`` table (see harness/matrix.py
for the row/axis semantics)."""

from __future__ import annotations

import pytest

from ...harness.matrix import (
    DEEPSEEK_V4_PRO,
    GLM_5_1_FP8,
    GLM_5_3,
    GLM_5_3_FLASH,
    GLM_5_3_FLASH_MXFP4,
    GLM_5_3_MXFP4,
    GPT_OSS,
    KIMI_K26_MXFP4,
    expand_cases,
)

# [enable, model, tp, ep, dp_attn] (+ optional opts dict). A tuple/list on an axis
# enumerates it (e.g. (True, False) runs both). MoE models can exercise ep.
CASES = [
    # gpt-oss-120b: tp2, ep on/off. On triton, not the default aiter backend: the
    # CK batch_prefill instance this case needs (page_size < kN0 over a >2GB KV
    # cache, gfx950) is absent from the aiter in the v0.5.17 base, so both TP ranks
    # raise "no matching kernel found" and the engine dies before serving. Drop this
    # once a base image carries the instance; aiter stays on for MoE either way.
    [
        True,
        GPT_OSS,
        2,
        True,
        False,
        {
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 1800,
            "args": ["--attention-backend", "triton"],
        },
    ],
    [
        True,
        KIMI_K26_MXFP4,
        4,
        True,
        True,
        {
            "env": {"SGLANG_USE_AITER": "1"},
            # Multithreaded weight load (forwarded verbatim to sglang's
            # ServerArgs / launch_server) to speed up loading the many shards.
            "args": [
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            "server_ready_timeout": 1800,
        },
    ],
    # DeepSeek-V4-Pro (MoE, tp8): --attention-backend dsv4 selects the DSv4 sparse
    # attention; the SGLANG_OPT_*/AITER env is its FP8 config. tp from the adapter.
    [
        False,
        DEEPSEEK_V4_PRO,
        8,
        False,
        False,
        {
            "args": [
                "--attention-backend",
                "dsv4",
                "--page-size",
                "256",
                "--disable-radix-cache",
                "--disable-shared-experts-fusion",
                "--swa-full-tokens-ratio",
                "0.15",
                "--mem-fraction-static",
                "0.90",
                "--chunked-prefill-size",
                "8192",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
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
            },
            "server_ready_timeout": 2400,
        },
    ],
    # GLM-5.1-FP8 (GlmMoeDsa = MLA + DSA lightning indexer, tp4). Minimal ON PURPOSE:
    # SGLang routes GlmMoeDsaForCausalLM through the DeepSeek MLA+DSA path and
    # auto-selects attention_backend=dsa / page_size=64 / tilelang / kv bf16 — do NOT
    # force the DSv4 flags (--attention-backend dsv4, --page-size 256), they fight the
    # auto-config. --reasoning-parser glm45 splits GLM reasoning_content; AITER on.
    # Verified 2026-07-23 single-node mix, temp=0: France->Paris/China->Beijing/2+2->4.
    # Long timeout covers the ~8-10 min silent tilelang-JIT + aiter-GEMM-tuning window.
    [
        True,
        GLM_5_1_FP8,
        4,
        False,
        False,
        {
            "args": [
                "--reasoning-parser",
                "glm45",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 1800,
        },
    ],
    # ---- GLM-5.3 series -----------------------------------------------------
    # All four are parked (enable=False) ON PURPOSE, and not because they are
    # unproven: each recipe below was brought up and smoke-checked on 8xMI355X.
    # They stay off because each needs ~300-700 GB of weights pre-staged and 4
    # GPUs for 10+ minutes of cold start, which no CI runner here has. Flipping
    # the first field to True is the whole activation step.
    #
    # TWO IMAGES. The Flash rows need deploy/docker/Dockerfile.sglang.glm53
    # (glm5_next exists in no released sglang; that file overlays sglang PR
    # #36607 at c821c425). The two big rows run on the stock
    # deploy/docker/Dockerfile.sglang image. Enabling a Flash row against the
    # wrong image fails at CONFIG LOAD with "model type `glm5_next` but
    # Transformers does not recognize this architecture" -- which names
    # transformers and invites the wrong fix; the missing piece is sglang.
    #
    # resolve_model() maps these ids to <INFERA_E2E_MODEL_DIR>/<id>. Pre-staged
    # trees here are flat (.../GLM-5.3-Flash-MXFP4), so symlink the vendor
    # prefix or the id falls back to a Hub pull of several hundred GB.
    [
        False,
        GLM_5_3_FLASH_MXFP4,
        4,
        False,
        False,
        {
            # --disable-shared-experts-fusion is LOAD-BEARING, not tuning. PR
            # #36607 opened the gfx950 branch of glm5_next's fusion gate
            # (glm5_next.py:1414) without carrying deepseek_v2.py:3069's
            # quant_blocks_shared_experts_fusion() guard, so this checkpoint's
            # BF16 shared expert is renamed into routed slot 288 of an
            # MXFP4-packed FusedMoE and weight load dies with
            # "size of tensor a (256) must match tensor b (512)".
            # Upstream #37268 is the same bug on NVFP4/NVIDIA, same workaround.
            # Health check: "Shared experts fusion optimization enabled." must
            # be ABSENT from the worker log.
            "args": [
                "--quantization",
                "quark",
                "--moe-runner-backend",
                "aiter",
                "--kv-cache-dtype",
                "bfloat16",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--disable-shared-experts-fusion",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--mm-feature-transport",
                "cpu",
                "--context-length",
                "65536",
                "--mem-fraction-static",
                "0.80",
                "--max-running-requests",
                "32",
                "--chunked-prefill-size",
                "4096",
                "--max-prefill-tokens",
                "16384",
            ],
            # SGLANG_USE_AITER gates #36607's AITER mHC dispatch. Without it the
            # server starts, answers correctly, and is 4.3-5.4x slower with
            # nothing in any log saying so -- grep the worker log for two
            # "AITER gfx950 mHC" lines per rank.
            # SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 is vendor-set for this
            # checkpoint and is absent from the FP8 recipe; not noise.
            "env": {"SGLANG_USE_AITER": "1", "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0"},
            # 650 s cold start observed on a node with a cold NFS cache.
            "server_ready_timeout": 3600,
        },
    ],
    [
        False,
        GLM_5_3_FLASH,
        4,
        False,
        False,
        {
            # FP8 original. No --quantization (config.json carries fp8) and the
            # triton MoE runner rather than aiter's FP4 path.
            # UNVERIFIED at time of writing: whether this checkpoint also needs
            # --disable-shared-experts-fusion. It does IFF its shared experts
            # are kept at a higher precision than its routed experts -- count
            # mlp.shared_experts .weight vs .weight_scale* in the safetensors
            # index. Add the flag if load dies in _load_w2/_load_w13 with a
            # 2:1 shape mismatch.
            "args": [
                "--moe-runner-backend",
                "triton",
                "--kv-cache-dtype",
                "bfloat16",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--context-length",
                "65536",
                "--mem-fraction-static",
                "0.85",
            ],
            "env": {"SGLANG_USE_AITER": "1"},
            "server_ready_timeout": 3600,
        },
    ],
    [
        False,
        GLM_5_3_MXFP4,
        4,
        True,
        False,
        {
            # glm_moe_dsa -- the GLM-5.2 code path, stock image. Quantization is
            # auto-detected from config.json; no --quantization needed.
            # The DSA-on-ROCm env block is MANDATORY on gfx950: without it the
            # model serves, returns 200s, and returns garbage, because the
            # sparse-attention indexer takes a path not ported to this arch.
            # infera.engine.sglang defaults SGLANG_OPT_USE_TOPK_V2 off on ROCm
            # (infera/engine/rocm_dsa_env.py); it is repeated here so a bare
            # launch_server run of this same row behaves identically.
            #
            # --disable-shared-experts-fusion is insurance rather than a fix:
            # glm4_moe.py:1174's gate only special-cases w4afp8 and would fuse
            # under quark, but this checkpoint's shared experts are themselves
            # MXFP4 (76 .weight / 75 .weight_scale, the odd one being the BF16
            # MTP layer 78, which is not loaded while MTP is off), so the
            # precondition is absent. Upstream #25261 shows this class failing
            # SILENTLY with wrong output rather than crashing, which is why it
            # is defaulted on. Drop it for a clean single-variable perf round.
            #
            # NOT the vendor card's --cuda-graph-max-bs 2 --max-running-requests
            # 2: that is a concurrency-2 accuracy configuration, not a
            # throughput one, and must not be copied into a benchmark arm.
            "args": [
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--moe-runner-backend",
                "aiter",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--context-length",
                "262144",
                "--mem-fraction-static",
                "0.80",
                "--chunked-prefill-size",
                "65536",
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "1",
                "SGLANG_OPT_USE_TOPK_V2": "0",
                "SGLANG_OPT_USE_JIT_NORM": "0",
            },
            "server_ready_timeout": 3600,
        },
    ],
    [
        False,
        GLM_5_3,
        4,
        True,
        False,
        {
            # FP8 original of the big model. Same code path as the MXFP4 row;
            # only the weights and the absent quantization flag differ. 704 GB
            # at TP4 leaves ~55 GB per GPU for KV at GMU 0.80 -- measured, not
            # estimated (max_total_num_tokens=1148288).
            "args": [
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--disable-custom-all-reduce",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--context-length",
                "262144",
                "--mem-fraction-static",
                "0.80",
                "--chunked-prefill-size",
                "65536",
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "1",
                "SGLANG_OPT_USE_TOPK_V2": "0",
                "SGLANG_OPT_USE_JIT_NORM": "0",
            },
            "server_ready_timeout": 3600,
        },
    ],
]


def sglang_mixed_params() -> list:
    """SGLang matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
