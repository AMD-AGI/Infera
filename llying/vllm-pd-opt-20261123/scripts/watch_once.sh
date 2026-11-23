#!/usr/bin/env bash
# Login-node watcher for one run_once.sbatch job of this directory: one "WATCH <job> <time> <event>" line per change.
#   watch_once.sh JOB ONCE_NAME RUN_DIR_NAME [RESULT_DIR]
# Events: job state; gate result; measurement start; score line; analyses done; failure keywords in the job's once log;
# crash signatures in the decode / prefill logs (counted, reported when they grow); no new decode steplog line for 10 min
# while measuring (possible stall). Reads only small files and greps logs; no heavy work on the login node.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
job=$1 once=$2 run=$3 res=${4:-}
D=$R/runs/$run
L=$R/logs/once-$job-$once.log
seen="" last_crash=0 stall_warned=0
say() { echo "WATCH $job $(date -u +%T) $*"; }
while true; do
    st=$(squeue -j "$job" -h -o "%T %N" 2>/dev/null); [ -z "$st" ] && st=GONE
    ev="$st"
    [ -n "$res" ] && grep -q "gate clean: yes" "$res/gate.txt" 2>/dev/null && ev+=" gate_ok"
    [ -n "$res" ] && grep -q "gate clean: no" "$res/gate.txt" 2>/dev/null && ev+=" GATE_FAIL"
    grep -qa "Phase profiling (profiling) started" "$D/agentx-c104/benchmark.log" 2>/dev/null && ev+=" measuring"
    grep -qa "score $run " "$R/logs/summary.out" 2>/dev/null && ev+=" scored"
    grep -qa "finish $run:" "$R/logs/summary.out" 2>/dev/null && ev+=" analysed"
    f=$(grep -a -c -i -E "NOT STARTED|deployment failed|smoke did not complete|engine start failed|overlay build / test failed|produced no table|gate clean: no" "$L" 2>/dev/null)
    [ "${f:-0}" -gt 0 ] && ev+=" FAIL_LINES=$f"
    [ "$ev" != "$seen" ] && { say "$ev"; seen=$ev; }
    c=$(cat "$D"/vpd-decode.stream.log "$D"/vpd-prefill.stream.log 2>/dev/null | grep -a -c -E "OUT_OF_RESOURCES|EngineCore encountered a fatal error|Segfault|died unexpectedly|RPC call to .* timed out")
    if [ "${c:-0}" -gt "$last_crash" ]; then say "CRASH_SIGNATURES=$c"; last_crash=$c; fi
    if [[ "$ev" == *measuring* && "$ev" != *scored* ]]; then
        newest=$(ls -t "$D"/steplog/steps-*.csv 2>/dev/null | head -1)
        if [ -n "$newest" ]; then
            age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
            if (( age > 600 && stall_warned == 0 )); then say "STALL? decode steplog idle ${age}s"; stall_warned=1; fi
            (( age < 120 )) && stall_warned=0
        fi
    fi
    [ "$st" = GONE ] && { say "job gone"; break; }
    sleep 60
done
