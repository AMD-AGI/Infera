#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# infera.server router (prefill node): the single OpenAI endpoint on :8000. It
# watches etcd, pairs the prefill and decode workers, and needs no worker list.
# Override: ETCD_ENDPOINT=... bash infera_1_server.sh
set -euo pipefail

HOST_IP="${HOST_IP:-${POD_IP:-$(ip -o -4 route get 1.1.1.1 | awk '{print $7}')}}"
ETCD_ENDPOINT="${ETCD_ENDPOINT:-${HOST_IP}:2379}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-8000}"
# python: the rust router is a separate cargo build and 1P1D is not router-bound.
ROUTER_BACKEND="${ROUTER_BACKEND:-python}"
# kv-aware is worth it even at 1P1D: rank-multiplexed workers fan out to one
# target per DP rank, so the policy still steers across 8 prefill and 8 decode
# ranks by prefix locality. It needs the legs to publish kv events (KV_EVENTS=1
# on infera_2/3); with events off it degrades to load-only, not to breakage.
ROUTER_POLICY="${ROUTER_POLICY:-kv-aware}"
LOG="${LOG:-$(dirname "$0")/infera_1_server.log}"

pkill -f "infera.server .*--port ${PORT}" 2>/dev/null || true
sleep 1

nohup python3 -m infera.server \
    --host 0.0.0.0 --port "$PORT" --router-backend "$ROUTER_BACKEND" \
    --etcd-endpoint "$ETCD_ENDPOINT" --router-tokenizer-path "$MODEL" \
    --discovery-backend etcd --request-transport http --kv-event-transport zmq \
    --router-policy "$ROUTER_POLICY" \
    --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0 \
    > "$LOG" 2>&1 &

for _ in $(seq 30); do
    sleep 1
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null \
    || { echo "[server] did not come up:"; tail -20 "$LOG"; exit 1; }
echo "[server] up on ${HOST_IP}:${PORT} — logs: $LOG"
