#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# infera.server router (node-0), SGLang stack. Override: ETCD_ENDPOINT=... bash infera_1_server.sh
set -euo pipefail

ETCD_ENDPOINT="${ETCD_ENDPOINT:-$(hostname -I | tr ' ' '\n' | awk -v p="${DATA_NET:-10.0.0.}" 'index($0,p)==1 {print; exit 0} END{exit 1}' || hostname -I | awk '{print $1}'):2379}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (local model dir, bind-mounted read-only)}"
: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
NAME="${NAME:-infera-sgl-server}"
ROUTER_BACKEND="${ROUTER_BACKEND:-rust}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host -v "$MODEL:$MODEL:ro" "$IMAGE" \
    python -m infera.server --host 0.0.0.0 --port 8000 --router-backend "$ROUTER_BACKEND" \
        --etcd-endpoint "$ETCD_ENDPOINT" --router-tokenizer-path "$MODEL" \
        --discovery-backend etcd --request-transport http --kv-event-transport zmq \
        --router-policy round-robin

echo "[server] $NAME up on :8000 — logs: docker logs -f $NAME"
