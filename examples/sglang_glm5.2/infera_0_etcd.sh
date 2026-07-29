#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# etcd for the SGLang PD test (prefill node). Shared registry: the workers take a
# lease and PUT their record, the router watches. Override: HOST_IP=... bash infera_0_etcd.sh
set -euo pipefail

HOST_IP="${HOST_IP:-${POD_IP:-$(ip -o -4 route get 1.1.1.1 | awk '{print $7}')}}"
ETCD_DIR="${ETCD_DIR:-/wekafs/llying/opt/etcd-v3.5.17-linux-amd64}"
# Local disk, not /wekafs: etcd fsyncs every write and hates a network filesystem.
DATA_DIR="${DATA_DIR:-/tmp/infera-etcd}"
LOG="${LOG:-$(dirname "$0")/infera_0_etcd.log}"

[[ -x "$ETCD_DIR/etcd" ]] || { echo "[etcd] not found: $ETCD_DIR/etcd — run install.sh"; exit 1; }
pkill -f "$ETCD_DIR/etcd" 2>/dev/null || true
[[ -n "${WIPE:-}" ]] && rm -rf "$DATA_DIR"
sleep 1

nohup "$ETCD_DIR/etcd" \
    --name infera-etcd --data-dir "$DATA_DIR" \
    --listen-client-urls "http://0.0.0.0:2379" \
    --advertise-client-urls "http://${HOST_IP}:2379" \
    --listen-peer-urls "http://127.0.0.1:2380" \
    --initial-advertise-peer-urls "http://127.0.0.1:2380" \
    --initial-cluster "infera-etcd=http://127.0.0.1:2380" \
    > "$LOG" 2>&1 &

for _ in $(seq 20); do
    sleep 1
    "$ETCD_DIR/etcdctl" --endpoints "http://${HOST_IP}:2379" endpoint health 2>/dev/null && break
done
"$ETCD_DIR/etcdctl" --endpoints "http://${HOST_IP}:2379" endpoint health || { tail -20 "$LOG"; exit 1; }
echo "[etcd] ready: --etcd-endpoint ${HOST_IP}:2379 — logs: $LOG"
