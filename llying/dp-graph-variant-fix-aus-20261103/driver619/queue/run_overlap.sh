#!/bin/bash
# DRIVER619-20261113, 11-17: run the queued single-node items on n02-25 (amdgpu 6.19) inside the user's hold job 36271
# (srun --overlap; its batch script only sleeps), the way kfd619_holder.sh runs them, until the queue is empty or the
# job stops running. NODE616 is a drained node outside the job: the items only pass it to --exclude.
unset ${!SLURM@}
K=/perf_apps/liyingli/bench_agentx/kfd619
export JOB=36271 NODE619=smci355-ccs-aus-n02-25 NODE616=smci355-ccs-aus-n01-21 K
say() { echo "$(date -u +%FT%TZ) kfd619 overlap $JOB: $*" | tee -a "$K/logs/holder.log"; }
say "start: queued items on $NODE619 inside job $JOB (srun --overlap)"
while item=$(ls "$K"/queue/*.sh 2>/dev/null | sort | head -1); [ -n "$item" ]; do
    squeue -h -j $JOB -o %T | grep -q RUNNING || { say "job $JOB not running: stop"; break; }
    name=$(basename "$item" .sh)
    mv "$item" "$K/running/" || continue
    say "run $name"
    bash "$K/running/$name.sh" > "$K/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo $rc > "$K/done/$name.rc"
    mv "$K/running/$name.sh" "$K/done/"
    say "done $name rc=$rc"
done
say "finished (queue: $(ls "$K"/queue/*.sh 2>/dev/null | wc -l) left)"
