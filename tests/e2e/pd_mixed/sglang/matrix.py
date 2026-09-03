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
    GPT_OSS,
    KIMI_K26_MXFP4,
    expand_cases,
)

# GLM-5.2 below is being brought up on the local MI300X fleet, where its weights
# are staged and its recipe was measured. Whether the gfx950 CI fleet has it at
# all is unconfirmed, so the skip claims only what is known — that nothing has
# run it there — rather than asserting a staging fact. Retiring it is one edit,
# and the row stays visible on both arches meanwhile.
_GFX950_UNMEASURED = {"skip": "brought up on gfx942; never run on the gfx950 CI fleet"}

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
            # Two measured facts, and the second is why the first was not worth
            # fixing. SGLang has no packed-MXFP4 expert kernel on gfx942, so this
            # checkpoint needs an engine-side FP4->FP8 dequant to load at all —
            # buildable, and built once: the dequant sits behind a branch
            # `_is_fp8_fnuz` never lets gfx942 reach, and hoisting it ahead made
            # the load log say "Dequantized FP4 expert weights to FP8". Then it
            # does not fit. Unpacked, the experts need 186.5 GiB a card at tp8
            # plus ~9.25 the runtime holds outside PyTorch, 195.8 against 191.98
            # usable — short before a byte of KV cache, measured twice. So the
            # patch is reverted rather than carried, and the dsv4 row SGLang
            # actually serves here is DeepSeek-V4-Flash-FP8 below.
            "gfx942": {
                "skip": "MXFP4 experts need an engine-side dequant SGLang cannot reach on gfx942, and unpacked they need 195.8 GiB/card against 191.98; run DeepSeek-V4-Flash-FP8 here, or Pro on vLLM which serves it packed",
            },
        },
    ],
    # DeepSeek-V4-Flash-FP8 (MoE, tp4) — the dsv4 cell SGLang can actually serve
    # on a 192 GiB card, and the reason is size rather than support. Same
    # architecture as the Pro row above (43 layers / 4096 hidden against 61 /
    # 7168) and block-FP8 throughout, so it never reaches the MXFP4 expert path
    # that skips Pro: 274 GiB, 68.5 a card at tp4, against Pro's 195.8 unpacked.
    #
    # tp4, not tp8. It fits either way (34.2 a card at tp8) and tp4 keeps the row
    # to half a node, which is what makes it affordable next to the tp8 GLM-5.2
    # row below. Every dimension divides: 64 attention heads, 64 index heads,
    # moe_intermediate_size 2048, 256 experts.
    #
    # The functional gfx942 knobs — `--attention-backend dsv4`,
    # `--disable-shared-experts-fusion` and four env vars — are ALSO applied by
    # infera.engine.dsv4_gfx942 at launch, set-if-unset. They are spelled out
    # anyway so this row reads as a complete recipe, and because the MTP flags
    # have to be: the speculation check reads `params.extra_args`, so flags that
    # only ever appear inside the worker leave it reporting requested=False and
    # unable to fail a run whose draft head silently did nothing.
    #
    # NOTHING HERE IS MEASURED ON THIS FLEET. The env block is the Pro row's,
    # which is the only dsv4-on-SGLang recipe that exists and was tuned on
    # MI325X; the SGLANG_OPT_USE_TILELANG_* offs are the same gfx942 avoidance as
    # the FlashMLA hack. MTP is on because the contract forces it for Flash — the
    # claim being that Flash's compressed-MQA decode kernel is broken on gfx942
    # and speculation routes around it, which no run here has checked either way.
    # A first green run against `--speculative-*` removed would be a finding.
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
            # fp8 behind SGLang's DSA auto-config, so gpt-oss's MXFP4 findings
            # do not carry; and this tp4 row has never been fitted against
            # MI300X's 192 GB per card (MI355X has 288).
            "gfx942": {"skip": "GLM-5.1-FP8 not measured on gfx942 yet"},
        },
    ],
    # GLM-5.2-FP8 (GlmMoeDsa, tp8 + dp-attention) — the knobs are the `aggregated`
    # arm of manual/recipes/glm5.2-fp8-gfx942.md, which is the one GLM-5.2 shape
    # measured on this hardware. Note this is a DIFFERENT model from the GLM-5.1
    # row above (78 layers, index_topk 2048, 154880 vocab), not a newer tag for
    # it, so both rows stand.
    #
    # tp8 is not a choice: ~700 GB of FP8 weights over 192 GB cards needs the whole
    # node, which also makes this the first mixed row to take one.
    #
    # MTP is on because the checkpoint ships the draft head (a 79th layer,
    # `model.layers.78.eh_proj.weight`, num_nextn_predict_layers 1) and the gfx942
    # image exists partly to serve it: its SGLang v0.5.16 base is pinned because
    # GLM-5.2 MTP needs sglang #30839 and GlmMoeDsaForCausalLMNextN, and on v0.5.15
    # the draft weight-load and PD warmup fail. 5/1/6 is the recipe's measured
    # depth — accept length 4.64 against a 4.00 break-even at 6.53 ms per draft
    # step. 7/1/8 misses break-even AND runs prefill out of activation memory, so
    # read it as a ceiling rather than a default.
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
]


def sglang_mixed_params() -> list:
    """SGLang matrix, expanded from :data:`CASES`."""
    return [pytest.param(p, id=p.id()) for p in expand_cases(CASES)]
