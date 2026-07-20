#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: tear down atom PD on both nodes (reap engines, remove etcd). why: free VRAM before next
# run (relaunch OOMs otherwise). how: pkill in container on each node; caller = user.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env PREFILL_NODE; require_env DECODE_NODE; CTR="${CTR:-dsv4_pd_atom}"
# Remove the container so its init reaps atom child procs; note: they otherwise hold VRAM + ports.
for h in "$PREFILL_NODE" "$DECODE_NODE"; do
  ssh -o StrictHostKeyChecking=no "$h" "docker rm -f $CTR repro-etcd 2>/dev/null; \
    pkill -9 -f 'infera.engine.atom' 2>/dev/null; true" || true
  log "torn down $h"
done
sleep 15   # let VRAM actually drain before any relaunch (removal returns before the GPU frees).
log "atom PD fully torn down (containers removed, VRAM freed)"
