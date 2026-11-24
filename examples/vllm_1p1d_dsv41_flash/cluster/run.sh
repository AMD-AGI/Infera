#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: subcommand dispatch shared by the cluster wrappers (they export the site values, then exec this).
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for v in PREFILL_NODE PREFILL_IP INFERA_IMAGE MODEL_MOUNT RDMA_IB_DEVICES; do
  case "${!v}" in *"<"*">"*) echo "[dsv41] ERROR: $v is still the placeholder '${!v}' — fill in cluster/$(basename "$0")" >&2; exit 1 ;; esac
done
cmd="${1:-}"; shift || true
case "$cmd" in
  up)       exec bash "$KIT/engine/up.sh" ;;
  smoke)    exec bash "$KIT/engine/smoke.sh" ;;
  accuracy) exec bash "$KIT/engine/accuracy.sh" ;;
  golden)   exec bash "$KIT/engine/golden.sh" golden ;;
  block)    exec bash "$KIT/engine/golden.sh" block ;;
  agentx)   [ $# -gt 0 ] || set -- 384; rc=0; for c in "$@"; do bash "$KIT/engine/agentx.sh" "$c" || rc=1; done; exit $rc ;;
  down)     exec bash "$KIT/engine/down.sh" ;;
  *) echo "usage: $(basename "$0") up | smoke | accuracy | golden | block | agentx CONC... | down" >&2; exit 1 ;;
esac
