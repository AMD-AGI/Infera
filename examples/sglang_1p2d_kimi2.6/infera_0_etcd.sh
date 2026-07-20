#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# etcd for the SGLang PD test (node-0). Override IP: ETCD_HOST_IP=10.0.0.1 bash infera_0_etcd.sh
set -euo pipefail

ETCD_HOST_IP="${ETCD_HOST_IP:-$(hostname -I | tr ' ' '\n' | awk -v p="${DATA_NET:-10.0.0.}" 'index($0,p)==1 {print; exit 0} END{exit 1}' || hostname -I | awk '{print $1}')}"
NAME="${NAME:-infera-sgl-etcd}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --net host "${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}" \
    etcd --advertise-client-urls "http://${ETCD_HOST_IP}:2379" \
         --listen-client-urls "http://0.0.0.0:2379" >/dev/null

sleep 3
docker exec "$NAME" etcdctl endpoint health || { docker logs "$NAME" 2>&1 | tail -20; exit 1; }
echo "[etcd] ready: --etcd-endpoint ${ETCD_HOST_IP}:2379"
