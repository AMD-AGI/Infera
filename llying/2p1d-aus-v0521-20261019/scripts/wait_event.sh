#!/usr/bin/env bash
# Block until something needs the operator: a new line in runtime/patrol/ALERT.log, a new "queue: step" line (the
# previous step finished), a new prepare_alloc.sh line, or TIMEOUT seconds. Prints what happened.
# Usage: ROUND=<r> JOB=<id> [TIMEOUT=6000] wait_event.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[[ -s "$R/operations/current_round" ]] && ROUND="$(cat "$R/operations/current_round")"
A="$R/runtime/patrol/ALERT.log"; Q="$R/runtime/runs/$ROUND/tune/queue.log"; PL="$R/logs/campaign.$JOB.log"
count() { local c; c="$(grep -c "$2" "$1" 2>/dev/null)"; echo "${c:-0}"; }
a0="$(count "$A" .)"; q0="$(count "$Q" 'queue: step')"; p0="$(count "$PL" "campaign:")"
end=$(( $(date +%s) + ${TIMEOUT:-6000} ))
while (( $(date +%s) < end )); do
    sleep 30
    a="$(count "$A" .)"; q="$(count "$Q" 'queue: step')"; p="$(count "$PL" "campaign:")"
    if (( a > a0 )); then echo "EVENT alert"; tail -n $((a - a0)) "$A"; exit 0; fi
    if (( q > q0 )); then echo "EVENT queue-step"; grep 'queue: step' "$Q" | tail -n $((q - q0)); exit 0; fi
    if (( p > p0 )); then echo "EVENT campaign"; grep "campaign:" "$PL" | tail -n $((p - p0)); exit 0; fi
done
echo "EVENT timeout"; tail -2 "$R/runtime/patrol/patrol.log"
