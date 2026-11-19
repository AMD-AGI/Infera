#!/usr/bin/env bash
# Wait for a scoring round's queue item to finish, then run every per-run analysis on this compute node (the node is
# released right after, so nothing is left for the login node to compute): post_run.sh into the branch's results
# directory, plus decode KV pressure, DP balance, crash / eviction counts, and the hang dumps and kernel logs.
#   finish_round.sh RC_FILE RUN_DIR_NAME RESULT_NAME     (RC_FILE relative to R)
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
cd "$R" || exit 0
rcf=$1 run=$2 name=$3
while [ ! -e "$rcf" ]; do sleep 30; ls "$(dirname "$rcf")" > /dev/null; done
sleep 60
d=runs/$run
O=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-p4d4-crusoe-20261117/results/$name
bash scripts/post_run.sh "$run" "$name"
mkdir -p "$O"
python3 scripts/decode_pressure.py "$d" 27301 60 > "$O/decode_pressure.txt" 2>&1
python3 scripts/dp_balance.py "$d" > "$O/dp_balance.txt" 2>&1
{
    for f in "$d"/vpd-decode.stream.log* "$d"/vpd-prefill.stream.log*; do
        [ -e "$f" ] || continue
        echo "$(basename "$f"): hipblas=$(zgrep -a -c HIPBLAS_STATUS "$f") out_of_resources=$(zgrep -a -c OUT_OF_RESOURCES "$f")" \
            "fatal=$(zgrep -a -c "EngineCore encountered a fatal error" "$f") shm_wait=$(zgrep -a -c "No available shared memory broadcast block" "$f")" \
            "jit=$(zgrep -a -c "JIT compilation during inference" "$f") preempt_lines=$(zgrep -a -c -i "preempt" "$f")"
    done
    echo "kernel log evict_process_worker lines: $(grep -a -c evict_process_worker "$d/dmesg-after.txt" 2>/dev/null)"
    grep -a evict_process_worker "$d/dmesg-after.txt" 2>/dev/null | tail -3
    cat "$d/container-state.txt" 2>/dev/null
    tail -3 "$d/memguard.log" 2>/dev/null
} > "$O/health.txt" 2>&1
[ -d "$d/hang" ] && cp -r "$d/hang" "$O/"
cp "$d/dmesg-after.txt" "$O/" 2>/dev/null
grep -a "$run" logs/summary.out | tail -2 > "$O/score.txt"
echo "$(date -u +%m%d-%T) finish $run: analyses in $O" >> logs/summary.out
exit 0
