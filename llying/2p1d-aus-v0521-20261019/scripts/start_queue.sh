#!/usr/bin/env bash
# Start the tuning queue driver on the login host unless one is already running (job 34439 has no ssh to its nodes;
# the node-side launcher is start_queue.sh.node-version). Usage: start_queue.sh ROUND=<r>
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
T="$R/runtime/runs/$ROUND/tune"; mkdir -p "$T"
running="$(ps -u "$(id -u)" -o pid=,args= | grep '[b]ash tune_queue.sh' | awk '{print $1}' || true)"
[[ -z "$running" ]] || { echo "queue driver already running: $running"; exit 0; }
set +u; set -a; source "$R/config/config.3p1d.sh"; set +a; set -u
cd "$HERE" && ROUND="$ROUND" QUEUE="$R/operations/tune_queue.tsv" setsid nohup bash tune_queue.sh >> "$T/queue.log" 2>&1 < /dev/null &
echo "queue driver started"
