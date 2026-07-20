#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: tear down sglang PD on all involved nodes (reap engines, remove etcd/container).
# why : free VRAM before next run/topology (relaunch OOMs otherwise). caller = user.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env PREFILL_NODE; require_env DECODE_NODE
CTR="${CTR:-dsv4_pd_sgl_mori}"; NODES="$PREFILL_NODE $DECODE_NODE ${P2_NODE:-}"
# Removing the container kills its `sleep infinity` init, which reaps the sglang child
# procs; note: they otherwise linger holding TCP ports (blocking the next launch) + VRAM.
for h in $NODES; do [ -n "$h" ] || continue
  ssh -o StrictHostKeyChecking=no "$h" "docker rm -f $CTR repro-etcd 2>/dev/null; \
    pkill -9 -f 'infera.engine.sglang' 2>/dev/null; true" || true
  log "torn down $h"
done
sleep 15   # let VRAM + ports actually drain before any relaunch (removal returns before they free).
log "sglang PD fully torn down (containers removed, ports + VRAM freed)"
