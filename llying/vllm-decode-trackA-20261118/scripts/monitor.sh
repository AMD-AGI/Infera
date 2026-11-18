#!/usr/bin/env bash
# Login-node watcher: prints one AGENT_LOOP_WAKE_vta line per event.
#   rc         a queue item finished whose name starts with L (long task), or any item finished non-zero
#   stale      the runner heartbeat (queue/RUNNER.hb) is older than 90 s (once per episode)
#   job        job 295840 is no longer running, or has less than 30 min left (once each)
#   heartbeat  every HB seconds (default 1800)
R=/home/liyingli/bench_agentx/vllm-trackA-20261118
Q=$R/queue
JOB=${JOB:-295840}
HB=${HB:-1800}
seen=$R/logs/monitor.seen
touch "$seen"
for f in "$Q"/*.rc; do [ -e "$f" ] && basename "$f" >> "$seen"; done
stale=0 gone=0 low=0 last_hb=$(date +%s)
wake() { echo "AGENT_LOOP_WAKE_vta $(date -u +%T) $*"; }
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
    left=$(squeue -j "$JOB" -h -t RUNNING -o "%L" 2>/dev/null)
    if [ -z "$left" ]; then
        (( gone == 0 )) && wake "{\"event\":\"job\",\"job\":$JOB,\"state\":\"not running\"}"
        gone=1
    else
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
