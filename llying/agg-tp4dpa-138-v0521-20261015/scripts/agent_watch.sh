#!/usr/bin/env bash
# Wake-ups for the agent's periodic inspection (run in the agent's shell with output notification).
# Prints AGENT_LOOP_TICK_agg every TICK_S seconds, and AGENT_LOOP_WAKE_agg at once when:
#   - a step driver fails or finishes;
#   - the newest engine log shows a new exception or fatal runtime error (a scheduler that dies on a
#     signal prints "Fatal Python error", not a Traceback: a6, 10-16 12:27);
#   - the running round's engine token counters stand still for STALL_S seconds while requests queue
#     (a6 hung with the client still printing progress, so client-log silence did not fire);
#   - the running round's client log has not advanced for STALL_S seconds.
# Usage: agent_watch.sh [TICK_S=900] [STALL_S=600]
set -u
RUNS="$(cd "$(dirname "$0")/.." && pwd)/runtime/runs"
tick="${1:-900}"
stall="${2:-600}"
events='FAILED|all [0-9]+ steps done'
fatal='Traceback|OutOfMemoryError|out of memory|Fatal Python error|HSA_STATUS_ERROR|Memory access fault'
declare -A seen
for f in "$RUNS"/steps.*.log; do [[ -e "$f" ]] && seen[$f]="$(grep -c -E "$events" "$f")"; done
elog=""; ecount=0; stalled=""; engine_stalled=""
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
        n="$(grep -a -c -E "$fatal" "$e")"
        if [[ "$e" != "$elog" ]]; then
            elog="$e"; ecount="$n"
        elif (( n > ecount )); then
            ecount="$n"
            wake "engine exception or fatal error in $e"
        fi
    fi
    a="$(ls -t "$RUNS"/*/tune/*/agentx.log 2>/dev/null | head -1)"
    if [[ -n "$a" && ! -e "$(dirname "$a")/bench.end" ]]; then
        age=$(( now - $(stat -c %Y "$a") ))
        if (( age > stall )) && [[ "$stalled" != "$a" ]]; then
            stalled="$a"
            wake "client log silent for ${age}s: $a"
        fi
        s="$(dirname "$a")/samples.jsonl"
        if [[ -e "$s" && "$engine_stalled" != "$s" ]]; then
            verdict="$(tail -n $(( stall / 10 + 1 )) "$s" | python3 -c '
import json, sys
rows = [json.loads(l) for l in sys.stdin if "\"m\"" in l]
if len(rows) < 2 or rows[-1]["t"] - rows[0]["t"] < float(sys.argv[1]) * 0.9:
    sys.exit()
tok = lambda m: sum(v for k, v in m.items() if k.startswith("realtime_tokens_total"))
queued = sum(v for k, v in rows[-1]["m"].items() if k.startswith(("num_queue_reqs", "num_running_reqs")))
if tok(rows[-1]["m"]) == tok(rows[0]["m"]) and queued > 0:
    print("stalled")' "$stall" 2>/dev/null)"
            if [[ "$verdict" == stalled ]]; then
                engine_stalled="$s"
                wake "engine token counters unchanged for ${stall}s with requests queued: $s"
            fi
        fi
    fi
    if (( now - last_tick >= tick )); then
        last_tick="$now"
        printf 'AGENT_LOOP_TICK_agg {"prompt":"periodic inspection of the aggregated run"}\n'
    fi
done
