#!/usr/bin/env bash
# Waits until the four prefill traces of one profile are copied out (prefill_profile_chain.sh prints "prof-c104 down"),
# then runs the GPU timeline and scheduler-CPU breakdown per rank and the cross-rank critical path.
# Usage: analyze_trace4_1028.sh CHAIN_LOG TRACE_DIR OUT_DIR
set -uo pipefail
log="$1" dir="$2" out="$3"
S=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/scripts
until grep -q "prof-c104 down" "$log"; do sleep 30; done
mkdir -p "$out"
t=()
for i in 0 1 2 3; do t+=("$(ls "$dir"/*-PP-$i.trace.json.gz)"); done
for i in 0 1 2 3; do
    python3 $S/prefill_gpu_timeline.py "${t[$i]}" >"$out/timeline-pp$i.txt" 2>&1 &
    python3 $S/prefill_sched_cpu.py "${t[$i]}" >"$out/sched-pp$i.txt" 2>&1 &
done
python3 $S/prefill_pp_critical.py "${t[@]}" --show 40 --why-ms 100 >"$out/pp-critical.txt" 2>&1
wait
echo "$(date -u +%T) trace analysis done"
