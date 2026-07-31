#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Aggregated GLM-5.2-FP8 SGLang server on one 8x MI325X (gfx942) node, in-container.
# MTP and DP_ATTENTION are independent switches; see README.md for the three variants.
# Override example: MODEL=/models/GLM-5.2-FP8 MTP=0 DP_ATTENTION=0 bash run_sglang.sh
set -euo pipefail

: "${MODEL:?set MODEL=/path/to/GLM-5.2-FP8 (local checkpoint dir, ~704 GiB)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MTP="${MTP:-1}"
DP_ATTENTION="${DP_ATTENTION:-1}"
DP_SIZE="${DP_SIZE:-$TP}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-4}"
LOG="${LOG:-$SCRIPT_DIR/run_sglang.log}"
PID_FILE="${PID_FILE:-$SCRIPT_DIR/run_sglang.pid}"

# gfx942 fails to build the draft kernels above 3 steps; catch it here rather than
# 20 minutes into the weight load.
[[ "$MTP" == 1 && "$MTP_STEPS" -gt 3 ]] \
    && { echo "ERROR: MTP_STEPS=$MTP_STEPS; gfx942 supports at most 3" >&2; exit 1; }

ARGS=()
# EAGLE against the checkpoint's own nextn layer; --disable-custom-all-reduce is
# required alongside it on this arch.
[[ "$MTP" == 1 ]] && ARGS+=(--disable-custom-all-reduce
                            --speculative-algorithm EAGLE
                            --speculative-num-steps "$MTP_STEPS"
                            --speculative-eagle-topk 1
                            --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS")
[[ "$DP_ATTENTION" == 1 ]] && ARGS+=(--enable-dp-attention --dp-size "$DP_SIZE")

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SAFETENSORS_FAST_GPU=1
export SGLANG_DSA_TRITON_PREFILL=1
export HSA_NO_SCRATCH_RECLAIM=1   # gfx942 firmware: distributed init aborts without it
export SGLANG_USE_AITER=1         # decides the DSA page size (64 vs 1) — pin it

nohup python3 -m sglang.launch_server \
    --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP" --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 131072 --mem-fraction-static "$MEM_FRAC" \
    --watchdog-timeout 1200 \
    "${ARGS[@]}" ${EXTRA_ARGS:-} \
    > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
DESC="TP$TP"
[[ "$DP_ATTENTION" == 1 ]] && DESC="$DESC + DP$DP_SIZE attention"
[[ "$MTP" == 1 ]] && DESC="$DESC + MTP($MTP_STEPS,$MTP_DRAFT_TOKENS)"
echo "[sglang] GLM-5.2-FP8 on :$PORT (pid $(cat "$PID_FILE")), $DESC"
echo "[sglang] cold start ~20 min with MTP, ~10 min without — logs -> $LOG"
echo "[sglang] ready when: curl -s 127.0.0.1:$PORT/health"
