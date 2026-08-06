#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: tear down the deployment on both nodes and wait for VRAM and ports to actually free.
# why : relaunching before that drains OOMs on a box that looks idle. The WAIT is the point.
# how : bash cluster/<your-cluster>.sh down
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env DECODE_NODE
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"

for h in "$PREFILL_NODE" "$DECODE_NODE"; do
  [ -n "$h" ] || continue
  # Removing the container kills its `sleep infinity` init, which reaps the engine children.
  # Left alone they linger holding VRAM and the KV-event port block, and the next launch dies
  # with "port_base at N is not available" — which reads as a port-allocation bug rather than
  # as leftover state.
  $SSH_CMD "$h" "docker rm -f $CTR glm52-etcd 2>/dev/null; \
    pkill -9 -f 'infera.engine.sglang' 2>/dev/null; true" >/dev/null 2>&1 || true
  log "torn down $h"
done

# Container removal returns before the GPU driver has released the memory.
sleep 20
log "checking VRAM released (a non-empty list means something is still holding it):"
for h in "$PREFILL_NODE" "$DECODE_NODE"; do
  echo "  --- $h ---"
  $SSH_CMD "$h" "rocm-smi --showpids 2>/dev/null | tail -8" || true
done
log "teardown complete"
