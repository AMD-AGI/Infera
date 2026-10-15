#!/usr/bin/env bash
# Start the tuning queue driver on the control node unless one is already running there
# (a repeated launch command once started a second driver that failed and clobbered queue.pid).
# Usage: start_queue.sh ROUND=<r>   (run from the login host)
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
T="$R/runtime/runs/$ROUND/tune"
running="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "ps -u \$(id -u) -o pid=,args= | grep '[b]ash tune_queue.sh' | awk '{print \$1}'" </dev/null)"
if [[ -n "$running" ]]; then
    echo "queue driver already running on crsuse2-m2m-137: $running"
    exit 0
fi
# Only the redirected command goes to the background: backgrounding a whole "a && b" list keeps
# ssh's stdout open in the subshell and the ssh never returns.
ssh -o BatchMode=yes -o ConnectTimeout=10 -n crsuse2-m2m-137 \
    "cd $R/scripts; ROUND=$ROUND QUEUE=$R/operations/tune_queue.tsv setsid nohup bash tune_queue.sh >> $T/queue.log 2>&1 < /dev/null &" </dev/null
echo "queue driver started"
