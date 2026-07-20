#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# SGLang PREFILL/producer (node-0): attn-DP4 + MoE-TP4 + fp8 KV + mooncake. Override: ETCD_ENDPOINT=... bash infera_2_sglang_prefill.sh
set -euo pipefail

HOST_IP="${HOST_IP:-$(hostname -I | tr ' ' '\n' | awk -v p="${DATA_NET:-10.0.0.}" 'index($0,p)==1 {print; exit 0} END{exit 1}' || hostname -I | awk '{print $1}')}"
ETCD_ENDPOINT="${ETCD_ENDPOINT:-${HOST_IP}:2379}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (local model dir, bind-mounted read-only)}"
: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
NAME="${NAME:-infera-sgl-prefill}"
GPUS="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-4}"; DP="${DP:-4}"
MAX_RUNNING="${MAX_RUNNING:-4096}"
MEM_FRAC="${MEM_FRAC:-0.85}"
CHUNK="${CHUNK:-131072}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --ipc=host --shm-size 32g \
    --cap-add=IPC_LOCK \
    -v "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}:/host-libionic/libionic.so:ro" \
    -v "$MODEL:$MODEL:ro" \
    -e HIP_VISIBLE_DEVICES="$GPUS" -e SGLANG_HOST_IP="$HOST_IP" -e HOST_IP="$HOST_IP" \
    -e MC_GID_INDEX=1 -e SGLANG_USE_AITER=1 \
    "$IMAGE" \
    python -m infera.engine.sglang \
        --model-path "$MODEL" --host 0.0.0.0 --port 30001 --advertise-host "$HOST_IP" \
        --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
        --request-transport http --no-enable-kv-events \
        --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
        --attention-backend aiter \
        --trust-remote-code --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING" \
        --chunked-prefill-size "$CHUNK" \
        --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
        --disaggregation-transfer-backend mooncake

LOG="${LOG:-$(dirname "$0")/infera_2_sglang_prefill.log}"
echo "[prefill] $NAME started — logs -> $LOG (Ctrl-C stops the view, NOT the container)"
docker logs -f "$NAME" 2>&1 | tee "$LOG"
