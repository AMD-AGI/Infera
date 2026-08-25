#!/usr/bin/env bash
# Launch the Infera kv-aware router on the prefill node.
# Run inside the engine container:
#   bash launch/launch_router.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

ROUTER_BACKEND="${ROUTER_BACKEND:-python}"
ROUTER_POLICY="${ROUTER_POLICY:-kv-aware}"
LOG="${LOG:-$LOG_DIR/router.log}"

pkill -f "infera.server .*--port ${ROUTER_PORT}" 2>/dev/null || true
pkill -f "infera-router.*port.*${ROUTER_PORT}" 2>/dev/null || true
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

echo "[router] ${ROUTER_POLICY}/${ROUTER_BACKEND} up on ${PREFILL_IP}:${ROUTER_PORT}; log=$LOG"
