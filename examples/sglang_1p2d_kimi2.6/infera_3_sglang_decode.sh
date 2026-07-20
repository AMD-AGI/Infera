#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# SGLang DECODE/consumer (node-1/node-2): attn-DP8 + MoE-EP1 (set EP>1 for mori a2a) + fp8 KV + mooncake. REQUIRED: ETCD_ENDPOINT=<node-0-data-ip>:2379 bash infera_3_sglang_decode.sh
set -euo pipefail

HOST_IP="${HOST_IP:-$(hostname -I | tr ' ' '\n' | awk -v p="${DATA_NET:-10.0.0.}" 'index($0,p)==1 {print; exit 0} END{exit 1}' || hostname -I | awk '{print $1}')}"
# etcd runs on node-0 (a different machine), so there is no sane local default here.
: "${ETCD_ENDPOINT:?set ETCD_ENDPOINT=<node-0-data-ip>:2379 (the node running infera_0_etcd.sh)}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (local model dir, bind-mounted read-only)}"
: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
NAME="${NAME:-infera-sgl-decode}"
GPUS="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"; DP="${DP:-8}"; EP="${EP:-1}"
MAX_RUNNING="${MAX_RUNNING:-4096}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MOE_ARGS=()
if [[ "$EP" != "1" && "$EP" != "0" ]]; then
    MOE_ARGS=(--ep-size "$EP" --moe-a2a-backend mori)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --ipc=host --shm-size 32g \
    --cap-add=IPC_LOCK \
    -v "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}:/host-libionic/libionic.so:ro" \
    -v "$MODEL:$MODEL:ro" \
    -e HIP_VISIBLE_DEVICES="$GPUS" -e SGLANG_HOST_IP="$HOST_IP" -e HOST_IP="$HOST_IP" \
    -e MC_GID_INDEX=1 -e SGLANG_USE_AITER=1 -e MORI_IB_GID_INDEX=1 -e RCCL_MSCCL_ENABLE=0 \
    "$IMAGE" \
    python -m infera.engine.sglang \
        --model-path "$MODEL" --host 0.0.0.0 --port 30002 --advertise-host "$HOST_IP" \
        --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
        --request-transport http --no-enable-kv-events \
        --tp-size "$TP" --dp-size "$DP" --enable-dp-attention ${MOE_ARGS[@]+"${MOE_ARGS[@]}"} \
        --attention-backend aiter \
        --trust-remote-code --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake

LOG="${LOG:-$(dirname "$0")/infera_3_sglang_decode_$(hostname -s).log}"
echo "[decode] $NAME started on $HOST_IP — logs -> $LOG (Ctrl-C stops the view, NOT the container)"
docker logs -f "$NAME" 2>&1 | tee "$LOG"
