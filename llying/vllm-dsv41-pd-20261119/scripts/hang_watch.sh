#!/usr/bin/env bash
# hang_watch.sh RUN_DIR [MAX_EPISODES]: on the host, watch the engine logs of one run for vLLM's shared-memory
# broadcast stall message (printed after 60 s without a step) and, when a new one appears, dump the Python stacks of
# every Python process in the engine container with py-spy (all of them plain, then the decode workers with --native),
# twice 60 s apart, plus GPU use. Output: RUN_DIR/pyspy/<episode>-<n>/. Stops after MAX_EPISODES (default 2).
set -uo pipefail
R=${1:?run dir}
MAX=${2:-2}
W=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119
CTR=llying-dsv41-eng
SPY=$W/wheels/extra/py-spy
PAT="No available shared memory broadcast block found"
count() { cat "$R"/decode.log "$R"/prefill*.log 2>/dev/null | grep -ac "$PAT"; }
dump() {
    local d=$1 pids p args
    mkdir -p "$d"
    docker exec "$CTR" ps -eo pid,ppid,stat,etime,pcpu,args > "$d/ps.txt" 2>&1
    rocm-smi --showuse --showmemuse --csv > "$d/rocm-smi.csv" 2>&1
    pids=$(awk 'NR > 1 && ($6 ~ /python/ || $6 ~ /^VLLM/) {print $1}' "$d/ps.txt")
    for p in $pids; do
        docker exec "$CTR" timeout 30 "$SPY" dump --pid "$p" > "$d/$p.txt" 2>&1 &
    done
    wait
    # Native frames (RCCL / HIP waits) for the decode workers only: slower, and they are the ones that stall.
    for p in $(awk 'NR > 1 && $0 ~ /Worker_DP|EngineCore_DP/ {print $1}' "$d/ps.txt"); do
        args=$(awk -v p="$p" '$1 == p' "$d/ps.txt")
        docker exec "$CTR" timeout 60 "$SPY" dump --native --pid "$p" > "$d/$p.native.txt" 2>&1 &
    done
    wait
    echo "$(date -u +%FT%TZ) dumped $(ls "$d" | grep -c '^[0-9]*\.txt$') processes" >> "$R/pyspy/watch.log"
}
mkdir -p "$R/pyspy"
base=$(count)
ep=0
echo "$(date -u +%FT%TZ) watching $R (baseline stall lines $base)" >> "$R/pyspy/watch.log"
while ((ep < MAX)); do
    sleep 15
    docker inspect "$CTR" >/dev/null 2>&1 || exit 0
    n=$(count)
    if ((n > base)); then
        ep=$((ep + 1))
        echo "$(date -u +%FT%TZ) stall seen ($n lines), episode $ep" >> "$R/pyspy/watch.log"
        dump "$R/pyspy/$ep-1"
        sleep 60
        dump "$R/pyspy/$ep-2"
        base=$(count)
        sleep 240
    fi
done
