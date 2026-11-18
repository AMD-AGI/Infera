#!/usr/bin/env bash
# Login-node watcher for the agent: prints one AGENT_LOOP_WAKE_vpd line per event.
#   rc         a queue item finished whose name starts with L (long task), or any item finished non-zero
#   stale      the runner heartbeat (queue/RUNNER.hb) is older than 90 s (once per episode)
#   job        the hold job is no longer running, has started running, or has less than 30 min left (once each episode)
#   heartbeat  every HB seconds (default 1800)
#   JOB=<id> or JOBNAME=<job name> (follows resubmissions), QDIR=<queue dir name under R> (defaults: 294199, queue)
R=/home/liyingli/bench_agentx/vllm-pd-20261117
Q=$R/${QDIR:-queue}
JOB=${JOB:-294199}
HB=${HB:-1800}
seen=$R/logs/monitor-${QDIR:-queue}.seen
touch "$seen"
for f in "$Q"/*.rc; do [ -e "$f" ] && basename "$f" >> "$seen"; done
stale=0 gone=0 low=0 last_hb=$(date +%s)
wake() { echo "AGENT_LOOP_WAKE_vpd $(date -u +%T) ${QDIR:-queue} $*"; }
while true; do
    ls "$Q" > /dev/null
    for f in "$Q"/*.rc; do
        [ -e "$f" ] || continue
        b=$(basename "$f")
        grep -qxF "$b" "$seen" && continue
        echo "$b" >> "$seen"
        rc=$(cat "$f" 2>/dev/null)
        name=${b#*-*-}; name=${name%.rc}
        if [[ "$name" == L* || "$rc" != 0 ]]; then wake "{\"event\":\"rc\",\"item\":\"${b%.rc}\",\"rc\":\"$rc\"}"; fi
    done
    if [ -e "$Q/RUNNER.hb" ] && [ ! -e "$Q/STOP" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$Q/RUNNER.hb") ))
        if (( age > 90 )); then
            (( stale == 0 )) && wake "{\"event\":\"stale\",\"age_s\":$age}"
            stale=1
        else
            stale=0
        fi
    fi
    if [ -n "${JOBNAME:-}" ]; then
        read -r jid left node <<< "$(squeue -u "$USER" -n "$JOBNAME" -h -t RUNNING -o "%i %L %N" 2>/dev/null | head -1)"
        [ -n "$jid" ] && JOB=$jid
    else
        read -r left node <<< "$(squeue -j "$JOB" -h -t RUNNING -o "%L %N" 2>/dev/null)"
    fi
    if [ -z "$left" ]; then
        (( gone == 0 )) && wake "{\"event\":\"job\",\"job\":\"${JOBNAME:-$JOB}\",\"state\":\"not running\"}"
        gone=1
    else
        (( gone == 1 )) && wake "{\"event\":\"job\",\"job\":$JOB,\"state\":\"running\",\"node\":\"$node\"}"
        gone=0
        secs=$(awk -v t="$left" 'BEGIN { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
            n = split(t, p, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + p[i]; print d * 86400 + s }')
        if (( secs < 1800 )); then
            (( low == 0 )) && wake "{\"event\":\"job\",\"job\":$JOB,\"left\":\"$left\"}"
            low=1
        fi
    fi
    now=$(date +%s)
    if (( now - last_hb >= HB )); then wake "{\"event\":\"heartbeat\"}"; last_hb=$now; fi
    sleep 30
done
