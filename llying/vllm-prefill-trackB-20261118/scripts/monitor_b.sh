#!/usr/bin/env bash
# Login-node watcher for track B: prints "TRACKB_WAKE ..." when a queue-n2 item finishes (rc file appears), when the
# runner heartbeat is older than 90 s, and once when job 294784 has less than 40 min left. Heartbeat line every 30 min.
Q=/home/liyingli/bench_agentx/vllm-pd-20261117/queue-n2
JOB=294784
seen=/home/liyingli/bench_agentx/vllm-trackB-20261118/logs/monitor.seen
touch "$seen"
warned_time=0 stale=0 n=0
while true; do
    ls "$Q" > /dev/null
    for f in "$Q"/*.rc; do
        [ -e "$f" ] || continue
        id=$(basename "$f" .rc)
        grep -qx "$id" "$seen" && continue
        echo "$id" >> "$seen"
        echo "TRACKB_WAKE done $id rc=$(cat "$f") $(date -u +%T)"
    done
    age=$(( $(date +%s) - $(stat -c %Y "$Q/RUNNER.hb" 2>/dev/null || echo 0) ))
    if (( age > 90 )); then
        (( stale == 0 )) && echo "TRACKB_WAKE runner heartbeat stale ${age}s $(date -u +%T)"
        stale=1
    else
        stale=0
    fi
    left=$(squeue -h -j $JOB -o %L 2>/dev/null)
    if [[ -z "$left" ]]; then
        echo "TRACKB_WAKE job $JOB gone $(date -u +%T)"; sleep 300; continue
    fi
    if [[ "$left" =~ ^([0-9]+):([0-9]+):[0-9]+$ ]] && (( 10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]} < 40 )) && (( warned_time == 0 )); then
        echo "TRACKB_WAKE job $JOB has $left left $(date -u +%T)"; warned_time=1
    fi
    (( n++ % 60 == 0 )) && echo "trackb monitor alive $(date -u +%T) job left $left"
    sleep 30
done
