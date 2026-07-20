#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# etcd service registry on the prefill node. Engines + server register here
# over the data-plane IP.
set -euo pipefail
ETCD_HOST_IP="${ETCD_HOST_IP:?set ETCD_HOST_IP to the prefill data-plane IP}"
ETCD_PORT="${ETCD_PORT:-12379}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-12380}"
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
ETCD_NAME="${ETCD_NAME:-pd-mori-etcd}"
docker rm -f "$ETCD_NAME" 2>/dev/null || true
echo "[etcd] starting on ${ETCD_HOST_IP}:${ETCD_PORT}"
docker run -d --name "$ETCD_NAME" --network=host --restart=unless-stopped "$ETCD_IMAGE" \
    etcd --data-dir=/var/lib/etcd \
    --listen-client-urls="http://0.0.0.0:${ETCD_PORT}" \
    --advertise-client-urls="http://${ETCD_HOST_IP}:${ETCD_PORT}" \
    --listen-peer-urls="http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}" \
    --initial-advertise-peer-urls="http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}" \
    --initial-cluster="default=http://${ETCD_HOST_IP}:${ETCD_PEER_PORT}"
for i in $(seq 1 30); do
    docker exec "$ETCD_NAME" etcdctl --endpoints="http://127.0.0.1:${ETCD_PORT}" endpoint health >/dev/null 2>&1 \
        && { echo "[etcd] healthy (${i}s)"; exit 0; }
    sleep 1
done
echo "[etcd] not healthy in 30s" >&2; docker logs "$ETCD_NAME" 2>&1 | tail -20 >&2; exit 1
