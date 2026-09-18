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

: "${CONTAINER_PREFIX:=glm52-pd-yihou-agentx-full}"

: "${YIHOU_BASE_CONFIG:=config.full.sh}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.base.sh"
