#!/usr/bin/env bash
# Start scripts/hang_watch.py in the background on the control node unless it is already running
# there (same idempotent pattern as start_queue.sh). Log: runtime/runs/<ROUND>/tune/hang_watch.log.
# Usage: start_hang_watch.sh ROUND=<r> [STALL=600] [INTERVAL=30]   (run from the login host)
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
T="$R/runtime/runs/$ROUND/tune"
running="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "ps -u \$(id -u) -o pid=,args= | grep '[h]ang_watch.py' | awk '{print \$1}'" </dev/null)"
if [[ -n "$running" ]]; then
    echo "hang watcher already running on crsuse2-m2m-137: $running"
    exit 0
fi
# Only the redirected command goes to the background: backgrounding a whole "a && b" list keeps
# ssh's stdout open in the subshell and the ssh never returns.
ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "cd $R/scripts; setsid nohup bash -c 'set -a; ROUND=$ROUND; source $R/config/config.3p1d.sh; set +a; exec python3 hang_watch.py $ROUND ${STALL:-600} ${INTERVAL:-30}' >> $T/hang_watch.log 2>&1 < /dev/null &" </dev/null
echo "hang watcher started"
