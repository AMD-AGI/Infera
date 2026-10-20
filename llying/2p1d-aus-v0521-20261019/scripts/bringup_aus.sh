#!/usr/bin/env bash
# Bring-up for a fresh ROUND, as a queue step: run_3p1d.sh all without benchmark points (preflight -> up at real
# acceptance -> census -> known-answer gate -> decode to simulated 3.61 + router -> smoke -> census), then the
# 192 x 8k-token KV transfer smoke through the router (every response non-empty, no decode transfer failure).
# Usage: bringup_aus.sh ROUND=<r>
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
POINTS=" " bash "$HERE/run_3p1d.sh" all || exit 1
set -a; source "$R/config/config.3p1d.sh"; set +a
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
python3 "$HERE/transfer_smoke.py" "http://$control_ip:$ROUTER_PORT" "$SERVED_MODEL" \
    "$RUNS/launch/server-logs/decode-0.after-realacc.log" 192 8000 > "$RUNS/transfer_smoke.txt" 2>&1
rc=$?
tail -5 "$RUNS/transfer_smoke.txt"
exit $rc
