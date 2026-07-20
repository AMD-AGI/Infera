#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Standalone SGLang server (not Infera, not PD-disaggregated): 4 GPUs, TP4.
# Override example: PORT=8000 HIP_VISIBLE_DEVICES=0,1,2,3 bash sglang_naive_engine.sh
set -euo pipefail

HOST_IP="${HOST_IP:-$(hostname -I | tr ' ' '\n' | awk -v p="${DATA_NET:-10.0.0.}" 'index($0,p)==1 {print; exit 0} END{exit 1}' || hostname -I | awk '{print $1}')}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (local model dir, bind-mounted read-only)}"
: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
NAME="${NAME:-sglang-kimi-engine}"
PORT="${PORT:-8000}"
GPUS="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-4}"
MAX_RUNNING="${MAX_RUNNING:-512}"
MEM_FRAC="${MEM_FRAC:-0.90}"
CHUNK="${CHUNK:-131072}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 32g \
    -v "$MODEL:$MODEL:ro" \
    -e HIP_VISIBLE_DEVICES="$GPUS" -e SGLANG_HOST_IP="$HOST_IP" -e HOST_IP="$HOST_IP" \
    -e SGLANG_USE_AITER=1 -e RCCL_MSCCL_ENABLE=0 \
    "$IMAGE" \
    python -m sglang.launch_server \
        --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
        --tp-size "$TP" \
        --attention-backend aiter \
        --trust-remote-code --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING" \
        --chunked-prefill-size "$CHUNK"

LOG="${LOG:-$(dirname "$0")/sglang_naive_engine_$(hostname -s).log}"
echo "[sglang-engine] $NAME started on $HOST_IP:$PORT, TP=$TP, GPUs=$GPUS — logs -> $LOG (Ctrl-C stops the view, NOT the container)"
docker logs -f "$NAME" 2>&1 | tee "$LOG"
