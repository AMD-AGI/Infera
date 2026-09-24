#!/usr/bin/env bash
# For each CONC in POINTS: cold-start the service, run AgentX, save logs, stop.
# Usage: scripts/sweep.sh [POINTS="80 112 144 192 256"] [DURATION=3600] [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"
dir="$(dirname "$0")"
sweep_id="${SWEEP_ID:-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
trap 'bash "$dir/down.sh"' EXIT

for conc in $POINTS; do
    bash "$dir/down.sh"
    bash "$dir/up.sh" CONC="$conc" RUN_ID="$sweep_id-c$conc"
    bash "$dir/agentx.sh" CONC="$conc" || echo "AgentX C$conc failed, see $(current_run)/agentx"
done
bash "$dir/down.sh"
RESULTS_DIR="$RESULTS_DIR" python3 "$dir/summarize.py" "$TMP_DIR/runs/$sweep_id"-c*
