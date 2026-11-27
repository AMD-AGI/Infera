#!/usr/bin/env bash
# Start one queue runner per node inside an existing allocation (the user's job 37014 has no batch script of ours).
# Run from the login node; srun stays in the background there (light) and the runners live on the compute nodes.
#   attach.sh JOBID
set -uo pipefail
job=${1:?job id}
W=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127
n=$(squeue -h -j "$job" -o %D)
nohup srun --jobid "$job" --overlap -N "$n" --ntasks-per-node=1 --kill-on-bad-exit=0 bash "$W/holder/holder.sh" \
    > "$W/logs/attach-$job.out" 2>&1 &
echo "attached runners to job $job ($n nodes), srun pid $!"
