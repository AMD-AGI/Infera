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
    DEEPSEEK_V4_FLASH_FP8,
    DEEPSEEK_V4_PRO,
    GLM_5_1_FP8,
    GLM_5_2_FP8,
    GLM_5_3,
    GLM_5_3_MXFP4,
    GPT_OSS,
    KIMI_K26_MXFP4,
    expand_cases,
)

# GLM-5.2 below was brought up on the local MI300X fleet — weights staged there,
# recipe measured there. Whether the gfx950 CI fleet has it at all is unconfirmed,
# so the reason claims only what is known (nothing has run it there) rather than
# asserting a staging fact; skip keeps the row visible on both arches meanwhile.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

# The GLM-5.3 rows are the mirror image: brought up on gfx950 only. The recipe
# below names gfx950 env and DSA backends throughout and has never been run on
# MI300X, so the row stays visible there with its reason rather than silently
# claiming coverage.
_GFX950_ONLY = {"skip": "brought up on gfx950; never run on the gfx942 CI fleet"}

# [enable, model, tp, ep, dp_attn] (+ optional opts dict). A tuple/list on an axis
# enumerates it (e.g. (True, False) runs both). MoE models can exercise ep.
CASES = [
    # gpt-oss-120b: tp2, ep on/off. Attention on triton, not aiter (which the env
    # below keeps for MoE): the CK batch_prefill instance this case needs (page_size
    # < kN0 over a >2GB KV cache, gfx950) is absent from the v0.5.17 base, so both TP
    # ranks raise "no matching kernel found" and die before serving. Drop when it lands.
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
            # Follow SGLang's native platform gate: gfx942 is not a supported
            # MXFP4 target for this checkpoint, so do not force a fallback.
            "gfx942": {
                "skip": "SGLang's upstream MXFP4 gate excludes gpt-oss-120b on gfx942",
            },
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
            # MXFP4 on aiter, the pairing gpt-oss above needed turned off here;
            # whether SGLANG_USE_AITER=0 serves this row too is one run away.
            # Skip until someone takes it — gfx950's knobs would only fail in a
            # way nobody has read.
            "gfx942": {"skip": "Kimi-K2.6 MXFP4 not measured on gfx942 yet"},
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
            # No packed-MXFP4 expert kernel on gfx942, so the checkpoint needs an
            # engine-side FP4->FP8 dequant to load at all. It is reachable — hoist
            # it past `_is_fp8_fnuz` — and still does not fit, so the patch stays
            # reverted: 186.5 GiB/card at tp8 + ~9.25 runtime = 195.8 vs 191.98 usable.
            "gfx942": {
                "skip": "MXFP4 experts need an engine-side dequant SGLang cannot reach on gfx942, and unpacked they need 195.8 GiB/card against 191.98; run DeepSeek-V4-Flash-FP8 here, or Pro on vLLM which serves it packed",
            },
        },
    ],
    # DeepSeek-V4-Flash-FP8 (MoE, tp4) — the dsv4 cell SGLang can actually serve on
    # a 192 GiB card, for size rather than support: same architecture as the Pro row
    # above (43 layers / 4096 hidden against 61 / 7168) and block-FP8 throughout, so
    # it never reaches the MXFP4 path that skips Pro. 274 GiB, 68.5 a card at tp4.
    #
    # tp4, not tp8. It fits either way (34.2 a card at tp8) and tp4 keeps the row
    # to half a node, which is what makes it affordable next to the tp8 GLM-5.2
    # row below. Every dimension divides: 64 attention heads, 64 index heads,
    # moe_intermediate_size 2048, 256 experts.
    #
    # These gfx942 knobs are ALSO applied set-if-unset by infera.engine.dsv4_gfx942
    # (dsv4 backend, --disable-shared-experts-fusion, four env vars); duplicated for
    # a complete recipe, and required for the MTP flags — the speculation check reads
    # `params.extra_args`, so worker-only flags cannot fail a no-op draft head.
    #
    # NOTHING HERE IS MEASURED ON THIS FLEET. The env block is the Pro row's — the
    # only dsv4-on-SGLang recipe that exists, tuned on MI325X. MTP is on because the
    # contract forces it for Flash, on the unchecked claim that Flash's compressed-MQA
    # decode kernel is broken on gfx942; a green run without `--speculative-*` is news.
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
                "--swa-full-tokens-ratio",
                "0.15",
                "--mem-fraction-static",
                "0.90",
                "--chunked-prefill-size",
                "8192",
                # The checkpoint declares max_position_embeddings 1048576 (yarn
                # x16 over 65536). Left uncapped SGLang sizes its KV pool for a
                # million tokens; 9472 is what the long-context probe needs and
                # what the rest of the matrix uses.
                "--context-length",
                "9472",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                # SGLang requires num_draft_tokens == num_steps + 1 for dsv4.
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
                "INFERA_ENGINE_READY_TIMEOUT": "3600",
            },
            # 274 GiB over NFS, then the draft head and graph capture. A third of
            # what GLM-5.2 reads for its 5400 s, and the JIT after it is the same
            # fixed cost either way.
            "server_ready_timeout": 3600,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
    # GLM-5.1-FP8 (GlmMoeDsa = MLA + DSA indexer, tp4). Minimal ON PURPOSE: SGLang
    # routes GlmMoeDsaForCausalLM through the DeepSeek MLA+DSA path, auto-selecting
    # dsa / page_size=64 / tilelang / kv bf16, and the DSv4 flags (dsv4, --page-size
    # 256) fight it. Verified 2026-07-23; timeout covers a ~8-10 min silent JIT+tune.
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
            # fp8 behind SGLang's DSA auto-config, so gpt-oss's MXFP4 findings
            # do not carry; and this tp4 row has never been fitted against
            # MI300X's 192 GB per card (MI355X has 288).
            "gfx942": {"skip": "GLM-5.1-FP8 not measured on gfx942 yet"},
        },
    ],
    # GLM-5.2-FP8 (GlmMoeDsa, tp8 + dp-attention) — knobs are the `aggregated` arm of
    # manual/recipes/glm5.2-fp8-gfx942.md, the one GLM-5.2 shape measured on this
    # hardware. A DIFFERENT model from the GLM-5.1 row above (78 layers, index_topk
    # 2048, 154880 vocab), not a newer tag for it, so both rows stand.
    #
    # tp8 is not a choice: ~700 GB of FP8 weights over 192 GB cards needs the whole
    # node, which also makes this the first mixed row to take one.
    #
    # MTP is on: the checkpoint ships the draft head (num_nextn_predict_layers 1) and
    # the image's SGLang v0.5.16 pin exists for it — v0.5.15 lacks #30839 /
    # GlmMoeDsaForCausalLMNextN and fails draft load + PD warmup. 5/1/6 is measured
    # (accept 4.64 vs 4.00 break-even at 6.53 ms/step); 7/1/8 misses it and OOMs.
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
                # SGLang's own tilelang DSA path, which the gfx942 image is
                # patched for; NOT the dsv4 backend — GLM's DSA is its own
                # (glm_moe_dsa, index_topk 2048) and the dsv4 flags fight its
                # auto-config, the trap the GLM-5.1 row above records.
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--mem-fraction-static",
                "0.85",
                "--max-running-requests",
                "128",
                # Aggregate, not per rank: dp-attention splits it CHUNK/DP, so
                # this is 1024/rank at dp8 — what the recipe's sweep chose.
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
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "6",
                # The draft iteration must not share the indexer with the target,
                # which is what the checkpoint's index_share_for_mtp_iteration
                # asks for by default.
                "--json-model-override-args",
                '{"index_share_for_mtp_iteration":false}',
            ],
            "env": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_DSA_TRITON_PREFILL": "1",
                "SAFETENSORS_FAST_GPU": "1",
                "HSA_NO_SCRATCH_RECLAIM": "1",
                # The engine's own patience, separate from the harness's below:
                # both have to cover the same cold start or the shorter one wins.
                "INFERA_ENGINE_READY_TIMEOUT": "5400",
            },
            # Weights land in ~3.5 min; the tilelang/aiter JIT and graph capture
            # after them print almost nothing for another ten or more, on every
            # fresh container. Don't tighten this against a warm run.
            "server_ready_timeout": 5400,
            "gfx950": _GFX950_UNMEASURED,
        },
    ],
    # ---- GLM-5.3 (big: glm_moe_dsa) -----------------------------------------
    # Parked (enable=False) ON PURPOSE, not because they are unproven: each needs
    # ~300-700 GB pre-staged and 4 GPUs for 10+ min of cold start. resolve_model()
    # maps ids to <INFERA_E2E_MODEL_DIR>/<id>; symlink the vendor prefix on a flat tree.
    [
        False,
        GLM_5_3_MXFP4,
        4,
        True,
        False,
        {
            # Quantization is auto-detected from config.json; no --quantization.
            # --disable-shared-experts-fusion is insurance, not a fix: the shared
            # experts are themselves MXFP4 and #25261 shows the class failing
            # SILENTLY with wrong output when shapes line up.
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
            # MANDATORY on gfx950: without this block the model serves, returns
            # 200s, and returns garbage, because the sparse-attention indexer
            # takes a path not ported to this arch. Mirrors the ROCm defaults in
            # infera/engine/rocm_dsa_env.py so a bare launch_server run matches.
            "env": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "1",
                "SGLANG_OPT_USE_TOPK_V2": "0",
                "SGLANG_OPT_USE_JIT_NORM": "0",
            },
            "server_ready_timeout": 3600,
            "gfx942": _GFX950_ONLY,
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
            "gfx942": _GFX950_ONLY,
        },
    ],
]


def sglang_mixed_params() -> list:
    """SGLang matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
