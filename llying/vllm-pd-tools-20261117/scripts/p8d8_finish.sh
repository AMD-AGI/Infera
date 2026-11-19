#!/usr/bin/env bash
# Per-run analyses of a P8D8 round, run on the job's node before it is released, into the branch's results directory:
# per-window throughput, decode KV pressure and DP balance (8 ranks), crash / eviction counts, smoke and hang dumps.
#   p8d8_finish.sh RUN_DIR_NAME RESULT_NAME [CONC=208]
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
cd "$R" || exit 0
run=$1 name=$2 conc=${3:-208}
d=runs/$run
O=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-p4d4-crusoe-20261117/results/$name
mkdir -p "$O"
exp=$d/agentx-c$conc/aiperf_artifacts/profile_export.jsonl
[ -e "$exp" ] && python3 scripts/window_tput.py "$exp" 300 16 > "$O/window_tput.txt" 2>&1
python3 scripts/decode_pressure.py "$d" 27301 60 > "$O/decode_pressure.txt" 2>&1
python3 scripts/dp_balance.py "$d" > "$O/dp_balance.txt" 2>&1
{
    for f in "$d"/vpd-decode.stream.log* "$d"/vpd-prefill.stream.log*; do
        [ -e "$f" ] || continue
        echo "$(basename "$f"): hipblas=$(zgrep -a -c HIPBLAS_STATUS "$f") out_of_resources=$(zgrep -a -c OUT_OF_RESOURCES "$f")" \
            "fatal=$(zgrep -a -c "EngineCore encountered a fatal error" "$f") notify_qp_fail=$(zgrep -a -c "Notification QP initialization failed" "$f")" \
            "reestablish=$(zgrep -a -c "Re-establish connection" "$f") kv_load_fail=$(zgrep -a -c "KV load failure" "$f")"
    done
    for f in "$d"/dmesg-after-*.txt; do echo "$(basename "$f"): evict_process_worker $(grep -a -c evict_process_worker "$f")"; done
    cat "$d/container-state.txt" 2>/dev/null
    tail -3 "$d/memguard.log" 2>/dev/null
} > "$O/health.txt" 2>&1
cp "$d"/smoke.out "$d"/workers.json "$d"/engine-*.cmd "$d"/container-state.txt "$d"/dmesg-after-*.txt "$O/" 2>/dev/null
cp "$d"/agentx-c$conc/agentx_conc*.json "$O/" 2>/dev/null
[ -d "$d/hang" ] && cp -r "$d/hang" "$O/"
grep -a "$run" logs/summary.out | tail -2 > "$O/score.txt"
echo "$(date -u +%m%d-%T) finish $run: analyses in $O" >> logs/summary.out
exit 0
