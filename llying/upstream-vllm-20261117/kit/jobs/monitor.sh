#!/usr/bin/env bash
# Login-node watcher for the plan jobs: one AGENT_WAKE_up line per event -- a job named lyl-up-* changes state
# (pending / running / gone), a step ends or is skipped (plan logs), a runner heartbeat goes older than 120 s --
# plus a heartbeat line every HB seconds (default 1800). Reads only small files.
#   bash jobs/monitor.sh
U=/home/liyingli/bench_agentx/upstream-20261119
HB=${HB:-1800}
declare -A state seen
wake() { echo "AGENT_WAKE_up $(date -u +%T) $*"; }
last=$(date +%s)
for f in "$U"/logs/plan-*.log; do [[ -e "$f" ]] && seen[$f]=$(wc -l < "$f"); done
while true; do
    now=$(squeue -u "$USER" -h -o "%i %j %T %N %M" 2>/dev/null | grep " lyl-up-")
    declare -A cur=()
    while read -r id name st node el; do
        [[ -z "$id" ]] && continue
        cur[$id]=1
        if [[ "${state[$id]:-}" != "$st" ]]; then wake "{\"event\":\"job\",\"job\":$id,\"name\":\"$name\",\"state\":\"$st\",\"node\":\"$node\"}"; state[$id]=$st; fi
    done <<< "$now"
    for id in "${!state[@]}"; do
        [[ -n "${cur[$id]:-}" ]] && continue
        wake "{\"event\":\"job\",\"job\":$id,\"state\":\"gone\"}"; unset "state[$id]"
    done
    unset cur
    for f in "$U"/logs/plan-*.log; do
        [[ -e "$f" ]] || continue
        n=$(wc -l < "$f")
        if (( n > ${seen[$f]:-0} )); then
            tail -n $(( n - ${seen[$f]:-0} )) "$f" | grep -E " end | skip | runner done| STOP| drain window over| stopping" | while read -r l; do
                wake "{\"event\":\"step\",\"log\":\"$(basename "$f")\",\"line\":\"$l\"}"
            done
            seen[$f]=$n
        fi
    done
    for hbf in "$U"/plans/*.hb; do
        [[ -e "$hbf" ]] || continue
        age=$(( $(date +%s) - $(cat "$hbf" 2>/dev/null || echo 0) ))
        if (( age > 120 && age < 600 )); then wake "{\"event\":\"stale\",\"hb\":\"$(basename "$hbf")\",\"age_s\":$age}"; fi
    done
    if (( $(date +%s) - last > HB )); then wake "{\"event\":\"heartbeat\"}"; last=$(date +%s); fi
    sleep 30
done
