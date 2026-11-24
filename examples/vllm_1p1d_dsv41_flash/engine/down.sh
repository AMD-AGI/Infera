#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: tear down the deployment and wait for VRAM to actually free.
# why : relaunching before that drains OOMs on a box that looks idle. The WAIT is the point.
# how : bash cluster/<your-cluster>.sh down
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"
require_env PREFILL_NODE; require_env DECODE_NODE

nodes=("$PREFILL_NODE"); [ "$DECODE_NODE" != "$PREFILL_NODE" ] && nodes+=("$DECODE_NODE")
for h in "${nodes[@]}"; do
  # Removing the container kills its `sleep infinity` init, which reaps the engines, the UMBP
  # server (and its DRAM pool) and the router.
  on "$h" "docker rm -f $CTR ${CTR}-etcd >/dev/null 2>&1; rm -rf $UMBP_DIR; true"
  log "torn down $h"
done
sleep 20
log "VRAM in use per GPU (all should read 0):"
for h in "${nodes[@]}"; do
  echo "  --- $h ---"
  on "$h" "rocm-smi --showmemuse 2>/dev/null | grep -E 'VRAM%' | sed 's/^/  /'" || true
done
