#!/usr/bin/env bash
# Wake-ups for the agent's periodic inspection (run in the agent's shell with output notification).
# Prints AGENT_LOOP_TICK_agg every TICK_S seconds, and AGENT_LOOP_WAKE_agg at once when a step driver
# fails or finishes, the newest engine log shows a new exception, or the running round's client log
# has not advanced for STALL_S seconds.
# Usage: agent_watch.sh [TICK_S=900] [STALL_S=600]
set -u
RUNS="$(cd "$(dirname "$0")/.." && pwd)/runtime/runs"
tick="${1:-900}"
stall="${2:-600}"
events='FAILED|all [0-9]+ steps done'
declare -A seen
for f in "$RUNS"/steps.*.log; do [[ -e "$f" ]] && seen[$f]="$(grep -c -E "$events" "$f")"; done
elog=""; ecount=0; stalled=""
last_tick="$(date +%s)"
wake() { printf 'AGENT_LOOP_WAKE_agg {"prompt":"%s"}\n' "$1"; }
while true; do
    sleep 60
    now="$(date +%s)"
    for f in "$RUNS"/steps.*.log; do
        [[ -e "$f" ]] || continue
        n="$(grep -c -E "$events" "$f")"
        if (( n > ${seen[$f]:-0} )); then
            seen[$f]="$n"
            wake "step driver: $(grep -E "$events" "$f" | tail -1 | tr '"' "'")"
        fi
    done
    e="$(ls -t "$RUNS"/*/launch/server-logs/mixed-0.log 2>/dev/null | head -1)"
    if [[ -n "$e" ]]; then
        n="$(grep -a -c -E 'Traceback|OutOfMemoryError|out of memory' "$e")"
        if [[ "$e" != "$elog" ]]; then
            elog="$e"; ecount="$n"
        elif (( n > ecount )); then
            ecount="$n"
            wake "engine exception in $e"
        fi
    fi
    a="$(ls -t "$RUNS"/*/tune/*/agentx.log 2>/dev/null | head -1)"
    if [[ -n "$a" && ! -e "$(dirname "$a")/bench.end" ]]; then
        age=$(( now - $(stat -c %Y "$a") ))
        if (( age > stall )) && [[ "$stalled" != "$a" ]]; then
            stalled="$a"
            wake "client log silent for ${age}s: $a"
        fi
    fi
    if (( now - last_tick >= tick )); then
        last_tick="$now"
        printf 'AGENT_LOOP_TICK_agg {"prompt":"periodic inspection of the aggregated run"}\n'
    fi
done
