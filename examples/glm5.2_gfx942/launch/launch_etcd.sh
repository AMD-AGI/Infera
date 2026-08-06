#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Start etcd on the prefill node, the registry both legs and the router discover
# each other through. Run on the HOST shell -- intentionally not in the engine
# container, which stays focused on infera/sglang and carries no etcd binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

HOST_IP="${ETCD_HOST_IP:-$PREFILL_IP}"
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"

docker rm -f "$ETCD_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$ETCD_CONTAINER" --net host "$ETCD_IMAGE" \
  etcd --advertise-client-urls "http://${HOST_IP}:2379" \
       --listen-client-urls "http://0.0.0.0:2379" >/dev/null

sleep 3
docker exec "$ETCD_CONTAINER" etcdctl endpoint health \
  || { docker logs "$ETCD_CONTAINER" 2>&1 | tail -40; exit 1; }
echo "[etcd] ready: ${HOST_IP}:2379"
