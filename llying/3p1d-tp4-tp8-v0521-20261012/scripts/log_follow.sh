#!/usr/bin/env bash
# Restart the server-log followers (bench-harness/follow_log.sh on each node) of every live engine
# container whose follower is gone, appending to runtime/runs/<ROUND>/launch/server-logs/<file> from
# just after the last timestamped line already there, so a dead follower leaves no gap. Followers
# started before follow_log.sh existed died for good on a failed write (/home full on 2026-10-13
# 10:42, 10-14 21:59, 10-15 05:03); follow_log.sh retries by itself. Safe to rerun.
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
    c="$CONTAINER_PREFIX-$instance"
    # Between retries follow_log.sh has no docker logs child, so look for either process.
    live="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" \
        "ps -eo args | grep -cE '^([d]ocker logs|bash .*[f]ollow_log\.sh) .* $c\$' || true")"
    if [[ "$live" != 0 ]]; then
        echo "$instance: follower alive on $node, left as is"
        continue
    fi
    ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" \
        "setsid nohup bash $BENCH_DIR/follow_log.sh $file $c > /dev/null 2>&1 < /dev/null &"
    echo "$instance: follower restarted on $node (resumes after the last timestamp in $(basename "$file"))"
done < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY")
