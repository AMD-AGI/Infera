#!/usr/bin/env bash
# Restart the server-log followers (nohup docker logs -f on each node) of every live engine
# container, appending to runtime/runs/<ROUND>/launch/server-logs/<file> from just after the
# last timestamped line already there, so a dead follower leaves no gap (followers die when a
# write fails, e.g. /home full on 2026-10-13 10:42).
# Usage: log_follow.sh ROUND=<r>   (log file per instance: <instance>.log, decode-0 uses DECODE_LOG)
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
LOGS="$RUNS/launch/server-logs"
DECODE_LOG="${DECODE_LOG:-decode-0.after-realacc.log}"
while IFS=$'\t' read -r index instance role node ip; do
    file="$LOGS/$instance.log"; [[ "$role" == decode ]] && file="$LOGS/$DECODE_LOG"
    since=""
    if [[ -s "$file" ]]; then
        last="$(tail -c 65536 "$file" | grep -a -oE '^[0-9]{4}-[0-9-]{5}T[0-9:]{8}\.[0-9]+Z' | tail -1)"
        since="$(python3 -c '
import sys
t = sys.argv[1].rstrip("Z"); base, frac = t.split(".")
n = int(frac.ljust(9, "0")[:9]) + 1
print("%s.%09dZ" % (base, n))' "$last")"
    fi
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
        "nohup docker logs --timestamps -f ${since:+--since $since} $CONTAINER_PREFIX-$instance >> $file 2>&1 < /dev/null &" </dev/null
    echo "$instance: follower restarted on $node (since ${since:-start})"
done < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY")
