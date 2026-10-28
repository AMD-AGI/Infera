#!/usr/bin/env bash
# After the E7 four-rank trace is copied out (chain log prints "prof-c104 down"): per-rank GPU timeline and scheduler
# breakdown, hop / NCCL timing and a ring timeline window, the same set as for the E1 trace, for a side-by-side comparison.
# Usage: analyze_trace4b_1028.sh CHAIN_LOG TRACE_PARENT OUT_DIR
set -uo pipefail
log="$1" parent="$2" out="$3"
S=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027/scripts
until grep -q "prof-c104 down" "$log"; do sleep 30; done
dir=$(ls -d "$parent"/prefill-*-p0 | tail -1)
mkdir -p "$out"
t=()
for i in 0 1 2 3; do t+=("$(ls "$dir"/*-PP-$i.trace.json.gz)"); done
for i in 0 1 2 3; do
    nice -n 10 python3 $S/prefill_gpu_timeline.py "${t[$i]}" >"$out/timeline-pp$i.txt" 2>&1 &
    nice -n 10 python3 $S/prefill_sched_cpu.py "${t[$i]}" >"$out/sched-pp$i.txt" 2>&1 &
done
wait
nice -n 10 python3 $S/prefill_pp_hops.py "${t[@]}" >"$out/pp-hops.txt" 2>&1 &
nice -n 10 python3 $S/prefill_pp_nccl.py "${t[@]}" >"$out/pp-nccl.txt" 2>&1 &
nice -n 10 python3 $S/nccl_kernel_shape.py "${t[0]}" >"$out/nccl-shape-pp0.txt" 2>&1 &
wait
nice -n 10 python3 $S/prefill_ring_timeline.py "${t[@]}" --from-step 30 --steps 8 >"$out/ring-timeline-30.txt" 2>&1
echo "$(date -u +%T) trace analysis done"
