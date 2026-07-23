#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Infera OpenAI-compatible router. Watches etcd, routes P/D.
set -euo pipefail
CONTAINER="${CONTAINER:-pd_mori_sgl}"
ETCD_HOST_IP="${ETCD_HOST_IP:?set ETCD_HOST_IP}"
ETCD_PORT="${ETCD_PORT:-12379}"
SERVER_PORT="${SERVER_PORT:-8000}"
ROUTER_POLICY="${ROUTER_POLICY:-round-robin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/PATH/TO/gpt-oss-120b}"
NATS_SERVER="${NATS_SERVER:?set NATS_SERVER, e.g. nats://<prefill-ip>:4222}"
REQUEST_TRANSPORT="${REQUEST_TRANSPORT:-nats}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERA_SRC="${INFERA_SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
LOG_FILE="${LOG_FILE:-${WORKSPACE}/logs/server_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[server] starting on 0.0.0.0:${SERVER_PORT} etcd=${ETCD_HOST_IP}:${ETCD_PORT} policy=${ROUTER_POLICY}"
docker exec -d "$CONTAINER" bash -lc "
    cd ${INFERA_SRC}
    exec >${LOG_FILE} 2>&1
    export PYTHONPATH=${INFERA_SRC}:\${PYTHONPATH:-}
    python3 -m infera.server --host 0.0.0.0 --port ${SERVER_PORT} \
        --discovery-backend etcd \
        --etcd-endpoint ${ETCD_HOST_IP}:${ETCD_PORT} --router-policy ${ROUTER_POLICY} \
        --request-transport ${REQUEST_TRANSPORT} \
        --nats-server ${NATS_SERVER} \
        --router-tokenizer-path ${TOKENIZER_PATH}
"
for i in $(seq 1 20); do
    sleep 1
    curl -fsS --max-time 2 "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1 \
        && { echo "[server] /health OK (${i}s)"; exit 0; }
done
echo "[server] /health not up in 20s" >&2; docker exec "$CONTAINER" tail -40 "$LOG_FILE" >&2 || true; exit 1
