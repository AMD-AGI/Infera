#!/usr/bin/env bash
# Block until the deployment answers THROUGH THE ROUTER, or exit non-zero.
#   ready = router /health 200
#         AND /v1/workers lists >= 1 worker with status "active"
#         AND /v1/models lists the served name.
# Exit 1 on timeout (E2E_KIT_READY_TIMEOUT), 2 if the engine process is gone.
# A wait that exits 0 when it gave up turns "the model never loaded" into "the
# benchmark measured nothing", found three stages later. So it never does.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

deadline=$(( $(date +%s) + E2E_KIT_READY_TIMEOUT ))
t0=$(date +%s)
while :; do
  if curl -sf -m5 "$KIT_ROUTER_URL/health" >/dev/null 2>&1; then
    workers=$(curl -sf -m10 "$KIT_ROUTER_URL/v1/workers" 2>/dev/null || true)
    models=$(curl -sf -m10 "$KIT_ROUTER_URL/v1/models" 2>/dev/null || true)
    if WORKERS="$workers" MODELS="$models" SERVED="$E2E_KIT_SERVED_NAME" python3 -c '
import json, os, sys
try:
    w = json.loads(os.environ["WORKERS"]); m = json.loads(os.environ["MODELS"])
except Exception:
    sys.exit(1)
ws = w.get("workers", []) if isinstance(w, dict) else w
active = [x for x in ws if isinstance(x, dict) and x.get("status") == "active"]
ids = [x.get("id") for x in m.get("data", [])]
sys.exit(0 if active and os.environ["SERVED"] in ids else 1)
'; then
      echo "[wait_ready] READY after $(( $(date +%s) - t0 ))s: $KIT_ROUTER_URL serves $E2E_KIT_SERVED_NAME"
      exit 0
    fi
  fi
  if docker ps -q --filter "name=^/${E2E_KIT_CONTAINER}\$" | grep -q . \
     && ! docker exec "$E2E_KIT_CONTAINER" pgrep -f infera.engine.sglang >/dev/null 2>&1; then
    echo "[wait_ready] engine process is gone in $E2E_KIT_CONTAINER; see ${KIT_LOG_DIR}/worker.log" >&2
    exit 2
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[wait_ready] TIMEOUT after ${E2E_KIT_READY_TIMEOUT}s: $KIT_ROUTER_URL not serving $E2E_KIT_SERVED_NAME" >&2
    exit 1
  fi
  sleep 5
done
