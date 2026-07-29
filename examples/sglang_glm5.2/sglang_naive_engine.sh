#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Standalone SGLang server (not Infera, not PD): the same GLM-5.2-FP8 + MTP +
# DP-attention recipe on one node. This is the aggregated baseline the PD numbers
# are compared against, and the fastest way to tell a model problem from a PD problem.
# Override: PORT=30000 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash sglang_naive_engine.sh
#
# Unlike the two PD legs this deliberately carries no patch guard: it is the A/B
# vehicle for the sglang patches, so it has to be able to run an unpatched tree.
set -euo pipefail

MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-30000}"
TP="${TP:-8}"; DP="${DP:-$TP}"
MEM_FRAC="${MEM_FRAC:-0.85}"
LOG="${LOG:-$(dirname "$0")/sglang_naive_engine.log}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export SGLANG_DSA_TRITON_PREFILL=1 SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1
# Decides the DSA page size (64 vs 1). Keep it pinned here so a comparison against
# the PD legs is actually apples-to-apples.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
# The image ships INT8; the single-node MTP recipe that was verified on this node
# (inference_glm5p2_sglang/run_sglang_mtp.sh) overrides it to INT4, so match that.
# Needs its own knob rather than a :- default on the real variable, which the image
# already sets — a :- default would silently inherit INT8. The PD legs take INT8.
export ROCM_QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE:-INT4}"

pkill -f "sglang.launch_server .*--port ${PORT}" 2>/dev/null || true
sleep 2

nohup python3 -m sglang.launch_server \
    --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
    --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --mem-fraction-static "$MEM_FRAC" --chunked-prefill-size 131072 \
    --watchdog-timeout 1200 --disable-custom-all-reduce \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    ${EXTRA_ARGS:-} \
    --weight-loader-prefetch-checkpoints \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    > "$LOG" 2>&1 &

echo "[naive] started on :${PORT}, TP=${TP} — logs -> $LOG"
tail -f "$LOG"
