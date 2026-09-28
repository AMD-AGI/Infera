#!/usr/bin/env bash
# Purpose: Alongside run_perf.sh driving ARM, take py-spy windows on the decode
#   schedulers at fixed offsets into the AgentX profiling phase.
# Usage: pyspy_decode.sh EVIDENCE_DIR ARM [OFFSET_S ...]   (default: 180 540)
# Artifacts: EVIDENCE_DIR/<ARM>-pyspy/ (see pyspy_window.sh)
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
E="$1"; arm="$2"; shift 2
offsets=("$@"); [[ ${#offsets[@]} -gt 0 ]] || offsets=(180 540)
cfg="$DIR/scripts/bench-harness/results/decrad/config.decrad.$arm.sh"
eval "$(bash -c "set -a; source '$cfg'; echo CONTAINER_PREFIX=\$CONTAINER_PREFIX")"
node=crsuse2-m2m-138
log="$E/$arm-agentx-c${CONC:-40}.log"

start=$(date +%s)
until grep -q "profiling\]" "$log" 2>/dev/null; do
    if grep -qE "$arm (launch failed|AgentX (done|FAILED))|busy before $arm" "$E/run.log" 2>/dev/null; then
        echo "$(date -u +%T) $arm ended before its profiling phase"; exit 1
    fi
    (( $(date +%s) - start < 14 * 3600 )) || { echo "$(date -u +%T) timed out"; exit 1; }
    sleep 10
done
t0=$(date +%s)
echo "$(date -u +%T) $arm profiling started"

k=0
for off in "${offsets[@]}"; do
    k=$((k + 1))
    now=$(date +%s)
    (( t0 + off > now )) && sleep $((t0 + off - now))
    echo "$(date -u +%T) window w$k"
    ssh -o BatchMode=yes "$node" bash "$DIR/scripts/pyspy_window.sh" \
        "$CONTAINER_PREFIX-decode-0" "$E/$arm-pyspy" "w$k" 90
done
echo "$(date -u +%T) done"
