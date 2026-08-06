#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch the infera-kvd daemon on the prefill node, BEFORE the prefill engine.
# Run INSIDE the engine container:  bash launch/launch_kvd.sh
#
# No-op unless KVD=1, so the bring-up sequence in the README can call it
# unconditionally. The prefill engine's socket probe refuses to start when the
# daemon is not answering, hence the order: kvd, then prefill.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

if [[ "$KVD" != "1" ]]; then
  echo "[kvd] KVD=0 -- no offload tier, GPU radix tree is the only cache"
  exit 0
fi

LOG="${LOG:-$LOG_DIR/kvd.log}"

mkdir -p "$(dirname "$KVD_SOCKET")" "$KVD_L3_DIR" \
  || { echo "[kvd] cannot create $(dirname "$KVD_SOCKET") or $KVD_L3_DIR" >&2; exit 1; }

pkill -f "infera\.kvd .*--socket ${KVD_SOCKET}( |$)" 2>/dev/null || true
# A stale socket file from a killed daemon still passes `test -S`, and the
# engine's probe then fails with Connection refused instead of waiting.
rm -f "$KVD_SOCKET"
sleep 1

# --use-tablespace is load-bearing: without it --long-path builds the legacy
# file-per-block region instead of the container-file tablespace (bounded file
# count, O_DIRECT, journal-recovered index).
nohup python3 -m infera.kvd \
  --socket "$KVD_SOCKET" \
  --max-bytes "$KVD_RAM_BYTES" \
  --long-path "$KVD_L3_DIR" --long-bytes "$KVD_LONG_BYTES" \
  --io-mode "$KVD_IO_MODE" \
  --use-tablespace --tablespace-pools "$KVD_TABLESPACE_POOLS" \
  > "$LOG" 2>&1 &

# Wait until the daemon answers an RPC, not merely until the socket file appears:
# the engine's own startup probe connects for real, so a weaker wait here would
# just move the failure into engine startup.
READY=0
for _ in $(seq 60); do
  if python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" >/dev/null 2>&1; then READY=1; break; fi
  sleep 1
done
[[ "$READY" == "1" ]] \
  || { echo "[kvd] daemon not answering on $KVD_SOCKET" >&2; tail -40 "$LOG"; exit 1; }

echo "[kvd] up: socket=$KVD_SOCKET ram=$KVD_RAM_BYTES l3=$KVD_L3_DIR ($KVD_LONG_BYTES); log=$LOG"
# The classifier picks O_DIRECT or buffered for L3, and a shared mount landing on
# buffered is the difference between reading L3 under the TTFT budget and inside
# it. Print the verdict rather than leaving it in the log.
grep -aE -m5 "io_mode|selfcheck" "$LOG" || true
