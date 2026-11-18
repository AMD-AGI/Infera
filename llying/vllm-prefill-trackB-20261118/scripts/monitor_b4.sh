#!/usr/bin/env bash
# Login-node watcher for the second track-B node: "TRACKB4_WAKE ..." when job $JOB changes state, when a queue-b item
# finishes, and when the queue-b runner heartbeat is older than 90 s while the job runs.
JOB=${1:?job id}
Q=/home/liyingli/bench_agentx/vllm-trackB-20261118/queue-b
seen=/home/liyingli/bench_agentx/vllm-trackB-20261118/logs/monitor-b4.seen
touch "$seen"
last= stale=0
while true; do
    st=$(squeue -h -j "$JOB" -o "%T %N" 2>/dev/null)
    [[ -z "$st" ]] && st=GONE
    [[ "$st" != "$last" ]] && echo "TRACKB4_WAKE job $JOB: $st $(date -u +%T)" && last=$st
    ls "$Q" > /dev/null 2>&1
    for f in "$Q"/*.rc; do
        [ -e "$f" ] || continue
        id=$(basename "$f" .rc)
        grep -qx "$id" "$seen" && continue
        echo "$id" >> "$seen"
        echo "TRACKB4_WAKE done $id rc=$(cat "$f") $(date -u +%T)"
    done
    if [[ "$st" == RUNNING* && -e "$Q/RUNNER.hb" ]]; then
        age=$(( $(date +%s) - $(stat -c %Y "$Q/RUNNER.hb") ))
        if (( age > 90 )); then (( stale == 0 )) && echo "TRACKB4_WAKE runner heartbeat stale ${age}s $(date -u +%T)"; stale=1; else stale=0; fi
    fi
    [[ "$st" == GONE ]] && break
    sleep 60
done
