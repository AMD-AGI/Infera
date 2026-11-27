#!/usr/bin/env bash
# Login-node watcher (light): wait for job JOBID to run, attach one queue runner per node, print one line per event.
#   watch_alloc.sh JOBID
job=${1:?job}
W=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127
last=""
while true; do
    st=$(squeue -h -j "$job" -o '%T %N %S' 2>/dev/null)
    [[ "$st" != "$last" ]] && echo "$(date -u +%T) job $job: ${st:-gone}" && last=$st
    case "$st" in
        RUNNING*) bash "$W/holder/attach.sh" "$job"; sleep 90; tail -5 "$W/holder/holder.log" 2>/dev/null; exit 0 ;;
        "") exit 1 ;;
    esac
    sleep 30
done
