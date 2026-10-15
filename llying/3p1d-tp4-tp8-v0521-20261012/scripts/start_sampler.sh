#!/usr/bin/env bash
# Start scripts/sampler.py in the background on the control node unless it is already running there
# (same idempotent pattern as start_queue.sh). Samples go to runtime/runs/<ROUND>/tune/samples.jsonl;
# the sampler's own stderr goes to node-local NODE_RW_ROOT/sampler.log, so a full /home cannot end it.
# Usage: start_sampler.sh ROUND=<r> [INTERVAL=60]   (run from the login host)
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
T="$RUNS/tune"
running="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "ps -u \$(id -u) -o pid=,args= | grep '[s]ampler.py' | awk '{print \$1}'" </dev/null)"
if [[ -n "$running" ]]; then
    echo "sampler already running on crsuse2-m2m-137: $running"
    exit 0
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "cd $R/scripts; setsid nohup bash -c 'set -a; ROUND=$ROUND; source $R/config/config.3p1d.sh; set +a; exec python3 sampler.py $T/samples.jsonl ${INTERVAL:-60}' >> $NODE_RW_ROOT/sampler.log 2>&1 < /dev/null &" </dev/null
echo "sampler started"
