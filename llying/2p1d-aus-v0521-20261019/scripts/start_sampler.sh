#!/usr/bin/env bash
# Start scripts/sampler.py on the login host unless already running (job 34439: no ssh to the nodes).
# Usage: start_sampler.sh ROUND=<r> [INTERVAL=60]
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set +u; set -a; source "$R/config/config.3p1d.sh"; set +a; set -u
T="$RUNS/tune"; mkdir -p "$T"
running="$(ps -u "$(id -u)" -o pid=,args= | grep '[s]ampler.py' | awk '{print $1}' || true)"
[[ -z "$running" ]] || { echo "sampler already running: $running"; exit 0; }
cd "$HERE" && setsid nohup python3 sampler.py "$T/samples.jsonl" "${INTERVAL:-60}" >> "$T/sampler.log" 2>&1 < /dev/null &
echo "sampler started"
