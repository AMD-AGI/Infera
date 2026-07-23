#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Stop the 1P1D stack on both nodes (engines via pkill inside container; etcd/server containers).
set -uo pipefail
PREFILL_NODE="${PREFILL_NODE:?set PREFILL_NODE}"
DECODE_NODE="${DECODE_NODE:?set DECODE_NODE}"
CONTAINER="${CONTAINER:-pd_mori_sgl}"
for n in "$PREFILL_NODE" "$DECODE_NODE"; do
    echo "[down] $n: stopping engine"
    ssh "$n" "docker exec ${CONTAINER} pkill -9 -f 'infera.engine.sglang' 2>/dev/null; docker exec ${CONTAINER} pkill -9 -f 'infera.server' 2>/dev/null" 2>/dev/null || true
done
ssh "$PREFILL_NODE" "docker rm -f pd-mori-etcd 2>/dev/null" 2>/dev/null || true
echo "[down] done"
