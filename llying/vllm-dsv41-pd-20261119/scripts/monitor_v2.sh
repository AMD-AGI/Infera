#!/usr/bin/env bash
# Login-node monitor (light: tail / stat / squeue only). Prints one "MON ..." line per event:
#   new summary.log lines, job state changes, hang-watch episodes, holder idle with an empty queue, and a running
#   AgentX whose runner.log has not grown for STALE_MIN minutes.
#   monitor.sh [JOBS...]   (default: 36738 36739)
W=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119
JOBS=("${@:-36738 36739}"); [ $# -eq 0 ] && JOBS=(36738 36739)
STALE_MIN=${STALE_MIN:-15}
declare -A st
lines=$(wc -l < "$W/logs/summary.log")
declare -A seen_watch stale_said
idle_said=""
while true; do
    n=$(wc -l < "$W/logs/summary.log")
    if ((n > lines)); then
        sed -n "$((lines + 1)),${n}p" "$W/logs/summary.log" | cut -c1-300 | sed 's/^/MON summary: /'
        lines=$n
    fi
    for j in "${JOBS[@]}"; do
        s="$(squeue -h -j "$j" -o '%T %N' 2>/dev/null)"; s=${s:-GONE}
        if [ "${st[$j]:-}" != "$s" ]; then echo "MON job $j: $s"; st[$j]=$s; fi
    done
    for f in "$W"/logs/*/pyspy/watch.log; do
        [ -e "$f" ] || continue
        c=$(grep -c "stall seen" "$f")
        if [ "${seen_watch[$f]:-0}" != "$c" ]; then
            ((c > 0)) && echo "MON hang-watch: $(dirname "$(dirname "$f")" | xargs basename): $(grep 'stall seen' "$f" | tail -1)"
            seen_watch[$f]=$c
        fi
    done
    for r in "$W"/holder/running/*.sh; do
        [ -e "$r" ] || continue
        item=$(basename "$r" .sh)
        d=$(ls -dt "$W"/logs/"$item"-* 2>/dev/null | head -1)
        [ -n "$d" ] || continue
        rl=$(ls -t "$d"/agentx-c*/runner.log 2>/dev/null | head -1)
        [ -n "$rl" ] || continue
        age=$(( ($(date +%s) - $(stat -c %Y "$rl")) / 60 ))
        if ((age >= STALE_MIN)) && [ -z "${stale_said[$rl]:-}" ]; then
            echo "MON STALE: $item runner.log unchanged for $age min"; stale_said[$rl]=1
        fi
    done
    q=$(ls "$W"/holder/queue/*.sh 2>/dev/null | wc -l); rn=$(ls "$W"/holder/running/*.sh 2>/dev/null | wc -l)
    running_jobs=$(for j in "${JOBS[@]}"; do [ "${st[$j]%% *}" = RUNNING ] && echo "$j"; done | wc -l)
    if ((q == 0 && rn < running_jobs)); then
        [ -z "$idle_said" ] && echo "MON IDLE: $running_jobs running holder(s), $rn running item(s), empty queue"
        idle_said=1
    else
        idle_said=""
    fi
    sleep 60
done
