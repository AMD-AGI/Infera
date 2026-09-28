#!/usr/bin/env bash
# Purpose: T2 -- AgentX concurrency sweep 40/56/72/96/128 in full mode with
#   SIMULATED MTP acceptance, prefill HiCache on, and upstream PR #37152 in the
#   image. Timing measurement; correctness is not what is being measured.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
#   One bring-up serves all five points -- agentx_bench.sh runs against the live
#   router, so the sweep does not need five launches.
# Artifacts: none; this file only defines shell variables.

# TWO FORCED SUBSTITUTIONS, not preferences.
# config.full.sh:87-89 requests the flydsl DSA backends and the aiter fused
# top-k. The pinned nightly in this image does not accept either -- verified
# first-hand against the image's own argparse choices during the C40 run
# (yihou/glm52-1p1d-samerail-c32-c40.packup_20260918, notes.md section 7).
# tilelang and the default sgl-kernel top-k are the substitutes that run.
: "${DSA_PREFILL_BACKEND:=tilelang}"
: "${DSA_DECODE_BACKEND:=tilelang}"
: "${DSA_TOPK_BACKEND:=sgl-kernel}"

# LEFT AS config.full.sh WRITES IT, deliberately:
#   JSON_MODEL_OVERRIDE_ARGS=""  -- the index_share_for_mtp_iteration=false
#     workaround stays removed. That is part of what "full" means. Note that the
#     previous run in this shape dropped 3 InvalidInferenceResultError, and that
#     it ran before the custom-all-reduce fix existed, so whether those errors
#     survive the fix is an open question this sweep incidentally tests.
#   DECODE_SIMULATE_ACC_LEN=3.61 -- simulated acceptance ON, per the task.
#     Timing here is comparable to other simulated runs and is NOT evidence that
#     this stack decodes correctly. T1 is where correctness is measured.
#   PREFILL_HICACHE=1, ratio 1.5  -- the reason PR #37152 is in the image.
#     HiCache host-pool release lags the request stream; slow memory return
#     between sweep points is expected, not a leak.

# NO --disable-custom-all-reduce ON THIS LEG. User decision, 2026-09-18, after a
# full resolved-server_args diff against the C40 run that completed 3600 s /
# 4128 requests / 0 errors on this exact shape.
#
# That diff had exactly ONE substantive entry:
#     disable_custom_all_reduce:  C40 False  ->  T2 True
# Everything else was byte-identical — dsa_{prefill,decode,topk}_backend, aiter
# fusion, max_running 128, EAGLE 5/1/6, json_model_override_args '{}'. Two
# independent runs carrying that flag died of a GPU memory access fault on
# decode (different GPUs, 32 min and 54 min in); C40 without it did not.
#
# CONSEQUENCE, accepted deliberately: decode output will be garbled, because
# this flag is the fix for exactly that. It does not affect what T2 measures —
# T2 runs under SGLANG_SIMULATE_ACC_LEN=3.61, so acceptance is forced and
# correctness is not the measurement. C40's numbers were taken the same way.
# Correctness lives in T1, which keeps the flag.
#
# Empty-but-SET, and config.yihou.base.sh uses "${DECODE_EXTRA_ARGS=...}" with a
# single dash so this survives instead of being reassigned.
: "${DECODE_EXTRA_ARGS=}"

# THE BASE IMAGE, NOT THE THIN-LAYER ONE. User decision, 2026-09-18, after a
# third GPU memory access fault — this one under a config whose resolved
# server_args were identical to C40's, which proves the engine arguments are not
# the trigger and that the remaining unaligned dimension is the IMAGE.
#
# C40 (3600 s, 4128 requests, 0 errors) ran `infera-sglang:v0519-yihou-0917`.
# All three faulting runs ran `…-nextnfix-hicache`, which adds exactly two things:
#   1. the GLM NextN shared-experts-fusion fix — flips num_fused_shared_experts
#      from 0 to non-zero on the draft, which is the discriminator in the
#      `_aiter_append` branch at layers/moe/topk.py:2219, whose own comment warns
#      that a double append "would write the shared id twice and evict a real
#      routed expert";
#   2. upstream PR #37152, the HiCache JIT copy-round widening — live here
#      because config.full.sh sets PREFILL_HICACHE=1, and NEVER executed in T1,
#      which has PREFILL_HICACHE=0.
# Either is a candidate; this run tests the image as a whole first.
#
# Consequence: the draft loses shared-experts fusion, so real MTP acceptance
# would be poor. Irrelevant to T2 — acceptance is forced to 3.61 here. T1 keeps
# the fixed image and is where acceptance is actually measured.
: "${IMAGE:=infera-sglang:v0519-yihou-0917}"

: "${CONTAINER_PREFIX:=glm52-pd-yihou-agentx-full}"

: "${YIHOU_BASE_CONFIG:=config.full.sh}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.base.sh"

# WORKAROUND for the decode-side DSA indexer GPU memory access fault.
# User decision, 2026-09-19. Full analysis: yihou/dsa.topk.indexer.bug.analysis.md
#
# GLM-5.2's own config.json sets index_share_for_mtp_iteration = TRUE, so an
# EMPTY --json-model-override-args does NOT mean "neutral" — it means the MTP
# draft reuses one indexer top-k selection across all draft steps, seeded at
# draft extend (eagle_worker_v2.py:288-303, 938-949, 1096-1103, 1152-1153).
# That path faulted in 4 of 5 runs; T1, which forced it false, did not.
# Upstream is unfixed: sglang #39517 and #37648 are open, and the hazard files
# are byte-identical between our build and origin/main.
#
# MUST BE SET AFTER THE SOURCE ABOVE. config.full.sh:90 assigns
# JSON_MODEL_OVERRIDE_ARGS="" unconditionally — a plain assignment, not
# ${VAR:-...} — so anything set before sourcing is silently overwritten.
#
# Cost: the indexer top-k is recomputed on each of the 4 in-loop draft-decode
# forwards instead of being reused. Compute only; the target verify recomputes
# its own selection, so served output is unchanged. Upstream #29787 puts the
# accept-length effect of index-share at <= +0.061.
#
# Not chosen, and why: SGLANG_DSA_FUSE_TOPK=0 also disables the suspect path but
# costs ~2.92x decode TPOT per PR #36714's own MI355X/gfx950 speed table.
# Switching the DSA backends to triton is the better long-term candidate (AMD's
# cookbook validates exactly our shape on triton, not tilelang) but changes more.
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
