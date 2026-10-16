#!/usr/bin/env bash
# Trigger sglang torch-profiler captures in the steady state of a running AgentX round: wait until the
# client log shows the profiling phase, then capture after DELAY1 s and again DELAY2 s later.
# Usage (on the worker node): prof_trigger.sh AGENTX_LOG ENGINE_URL OUT_DIR [DELAY1=480] [DELAY2=600] [STEPS=30]
set -u
log="$1"; url="$2"; out="$3"; d1="${4:-480}"; d2="${5:-600}"; steps="${6:-30}"
say() { echo "$(date -u +%FT%TZ) $*"; }
until grep -a -q ' srv ' "$log" 2>/dev/null; do sleep 20; done
say "profiling phase seen; first capture in ${d1}s"
sleep "$d1"
for i in 1 2; do
    mkdir -p "$out/cap$i"
    body="{\"output_dir\":\"$out/cap$i\",\"num_steps\":$steps,\"profile_by_stage\":true,\"profile_prefix\":\"cap$i\",\"with_stack\":false,\"record_shapes\":false}"
    say "capture $i: $(curl -sS --max-time 120 -X POST "$url/start_profile" -H 'Content-Type: application/json' -d "$body" | tr -d '\n')"
    curl -fsS --max-time 10 "$url/metrics" | grep -E '^sglang:cuda_graph_passes_total' > "$out/cap$i.passes.start" 2>/dev/null
    (( i == 1 )) && sleep "$d2"
done
sleep 300
curl -fsS --max-time 10 "$url/metrics" | grep -E '^sglang:cuda_graph_passes_total' > "$out/cap2.passes.end" 2>/dev/null
say "done; files: $(ls "$out"/cap*/ 2>/dev/null | wc -l)"
