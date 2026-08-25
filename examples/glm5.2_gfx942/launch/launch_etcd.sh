#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Start etcd on the prefill node, the registry both legs and the router discover
# each other through. Run on the HOST shell -- intentionally not in the engine
# container, which stays focused on infera/sglang and carries no etcd binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
require_ips

HOST_IP="${ETCD_HOST_IP:-$PREFILL_IP}"
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"

# Take the port from ETCD_ENDPOINT rather than hard-coding 2379, which is what the
# engines and the router dial. Hard-coding it here means a non-default
# ETCD_ENDPOINT points them at a port nothing listens on, and that only surfaces
# after the weights load. The peer port has to move with it: --net host puts both
# on the node, and a node already running something on 2379/2380 -- a k8s control
# plane is the usual one -- takes the client port down with the peer port even
# though PD never uses peer at all.
ETCD_PORT="${ETCD_ENDPOINT##*:}"
# A bare host is a legal ETCD_ENDPOINT -- the engines and the router accept
# `host`, `host:port` or a URL and fill in 2379 themselves -- so assume the port
# they will dial rather than failing the peer arithmetic below on the host.
[[ "$ETCD_PORT" =~ ^[0-9]+$ ]] || ETCD_PORT=2379
ETCD_PEER_PORT="${ETCD_PEER_PORT:-$((ETCD_PORT + 1))}"
PEER_URL="http://127.0.0.1:${ETCD_PEER_PORT}"

docker rm -f "$ETCD_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$ETCD_CONTAINER" --net host "$ETCD_IMAGE" \
  etcd --advertise-client-urls "http://${HOST_IP}:${ETCD_PORT}" \
       --listen-client-urls "http://0.0.0.0:${ETCD_PORT}" \
       --listen-peer-urls "$PEER_URL" \
       --initial-advertise-peer-urls "$PEER_URL" \
       --initial-cluster "default=${PEER_URL}" >/dev/null

sleep 3
docker exec "$ETCD_CONTAINER" etcdctl --endpoints "http://127.0.0.1:${ETCD_PORT}" endpoint health \
  || { docker logs "$ETCD_CONTAINER" 2>&1 | tail -40; exit 1; }
echo "[etcd] ready: ${HOST_IP}:${ETCD_PORT}"
