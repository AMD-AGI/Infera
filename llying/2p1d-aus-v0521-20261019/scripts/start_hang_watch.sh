#!/usr/bin/env bash
# Start scripts/hang_watch.py on the login host unless already running (job 34439: no ssh to the nodes).
# Usage: start_hang_watch.sh ROUND=<r> [STALL=600] [INTERVAL=30]
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set +u; set -a; source "$R/config/config.3p1d.sh"; set +a; set -u
T="$R/runtime/runs/$ROUND/tune"; mkdir -p "$T"
running="$(ps -u "$(id -u)" -o pid=,args= | grep '[h]ang_watch.py' | awk '{print $1}' || true)"
[[ -z "$running" ]] || { echo "hang watcher already running: $running"; exit 0; }
cd "$HERE" && setsid nohup python3 hang_watch.py "$ROUND" "${STALL:-600}" "${INTERVAL:-30}" >> "$T/hang_watch.log" 2>&1 < /dev/null &
echo "hang watcher started"
