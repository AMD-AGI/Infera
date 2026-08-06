#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch the Infera kv-aware router on the prefill node.
# Run INSIDE the engine container:  bash launch/launch_router.sh
#
# The router is a separate process that rediscovers both legs from etcd, so it can
# be restarted on its own -- the engines never move.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

LOG="${LOG:-$LOG_DIR/router.log}"

# `rust` execs the infera-router binary over this python process. infera.server
# checks the supported subset before the exec and fails with a pointer to
# --router-backend python; everything passed here is inside it. One thing worth
# naming: the rust data plane ignores --kv-event-transport and always subscribes
# over ZMQ, which is the transport configured here anyway.
if [[ "$ROUTER_BACKEND" == "rust" ]]; then
  command -v infera-router >/dev/null 2>&1 || [[ -n "${INFERA_ROUTER_BIN:-}" ]] \
    || { echo "[router] ROUTER_BACKEND=rust but no infera-router on PATH." >&2
         echo "         This image builds it to /usr/local/bin; set" >&2
         echo "         INFERA_ROUTER_BIN if yours does not." >&2; exit 1; }
fi

# After the exec the process is named infera-router and no longer matches the
# python pattern, so kill both.
pkill -f "infera.server .*--port ${ROUTER_PORT}" 2>/dev/null || true
pkill -f "infera-router.*--port ${ROUTER_PORT}" 2>/dev/null || true
sleep 1

nohup python3 -m infera.server \
  --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend "$ROUTER_BACKEND" \
  --etcd-endpoint "$ETCD_ENDPOINT" --router-tokenizer-path "$MODEL" \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq \
  --router-policy "$ROUTER_POLICY" \
  --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0 \
  > "$LOG" 2>&1 &

for _ in $(seq 60); do
  sleep 1
  curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null && break
done
curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null \
  || { echo "[router] did not come up"; tail -40 "$LOG"; exit 1; }

# kv-aware needs the tokenizer to compute block hashes. Without it the router does
# not fail -- it warns once and routes on load alone, which looks like a healthy
# router that has quietly stopped doing the thing you deployed it for.
if [[ "$ROUTER_POLICY" == "kv-aware" ]] \
   && grep -qa "degrades to pure load balancing" "$LOG" 2>/dev/null; then
  echo "[router] kv-aware degraded to load balancing -- tokenizer not loaded from $MODEL" >&2
  grep -a "degrades to pure load balancing" "$LOG" >&2
  exit 1
fi

echo "[router] ${ROUTER_POLICY}/${ROUTER_BACKEND} up on ${PREFILL_IP}:${ROUTER_PORT}; log=$LOG"
