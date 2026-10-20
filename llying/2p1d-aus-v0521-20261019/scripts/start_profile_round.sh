#!/usr/bin/env bash
# Queue step: start a detached watcher (login host) that takes one decode torch-profiler capture in the steady state of an
# upcoming round, for the decode warm-up drift study (RESULT §2.12): wait until the round's agentx.log shows the AgentX
# profiling phase, sleep DELAY s, POST /start_profile to decode-0 (STEPS forward steps, by stage), wait, copy the traces
# back to runtime/runs/<ROUND>/profiles/<VARIANT>/ and run trace_breakdown.py on them. Returns immediately.
# Usage: start_profile_round.sh ROUND=<r> VARIANT=<v> CONC=<c> [DELAY=900] [STEPS=40]
set -euo pipefail
for a in "$@"; do export "$a"; done
: "${ROUND:?}" "${VARIANT:?}" "${CONC:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set +u; set -a; source "$R/config/config.3p1d.sh"; set +a; set -u
OUT="$RUNS/profiles/$VARIANT"; mkdir -p "$OUT"
read -r index instance role node ip < <(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' '$3=="decode"' | head -1)
url="http://$ip:$((ENGINE_PORT_BASE + index * ENGINE_PORT_STRIDE))"
remote="$NODE_RW_ROOT/profiles/$ROUND/$VARIANT"
log="$RUNS/tune/$VARIANT-c$CONC/agentx.log"
setsid nohup bash -c "
    say() { echo \"\$(date -u +%FT%TZ) \$*\"; }
    say 'waiting for the profiling phase in $log'
    until grep -a -q 'Phase profiling (profiling) started' '$log' 2>/dev/null; do sleep 30; done
    say 'profiling phase seen; capture in ${DELAY:-900} s'
    sleep ${DELAY:-900}
    ssh -n $node 'mkdir -p $remote'
    body='{\"output_dir\":\"$remote\",\"num_steps\":${STEPS:-40},\"profile_by_stage\":true,\"profile_prefix\":\"$VARIANT\",\"with_stack\":false,\"record_shapes\":false}'
    say \"start_profile: \$(curl -sS --max-time 300 -X POST '$url/start_profile' -H 'Content-Type: application/json' -d \"\$body\" | tr -d '\n')\"
    sleep 180
    ssh -n $node 'tar -C $remote -cf - . | base64 -w0' | base64 -d | tar -C '$OUT' -xf - && say \"copied \$(ls '$OUT' | wc -l) files\"
    python3 '$R/scripts/trace_breakdown.py' \$(ls '$OUT'/*.json.gz 2>/dev/null | head -8) > '$OUT/breakdown.txt' 2>&1 && say 'breakdown written'
" >> "$OUT/watch.log" 2>&1 < /dev/null &
echo "profile watcher started for $VARIANT (decode $url, delay ${DELAY:-900}s, ${STEPS:-40} steps) -> $OUT"
