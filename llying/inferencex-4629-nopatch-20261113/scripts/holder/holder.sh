#!/usr/bin/env bash
# Two-node holder for the InferenceX#4629 rocm10 check (infx-aus2 tree). Holds the nodes with the
# resources srtctl's own sbatch asks for and runs queued items inside this allocation, one at a
# time, by name. An item is a bash script, typically the sbatch_script.sh srtctl rendered for one
# matrix point, so do_sweep and everything below it run as in CI's job.
#   queue: $H/queue/<NN>-<label>.sh   running: $H/running/   done: $H/done/<name>.sh + .rc
#   logs:  $H/logs/<name>.log         stop: touch $H/STOP (exit after the current item)
# Submit: sbatch --exclude="$(paste -sd, /apps/liyingli/jobs/exclude_nodes.txt)" holder.sh
#SBATCH --job-name=llying-infx4629
#SBATCH --partition=Compute-Multinode
#SBATCH --account=xiewen12
#SBATCH --qos=batch
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/apps/liyingli/holder-infx/holder-%j.out
set -uo pipefail
H=/apps/liyingli/holder-infx
mkdir -p "$H"/{queue,running,done,logs}
say() { echo "$(date -u +%FT%TZ) holder $SLURM_JOB_ID: $*" | tee -a "$H/logs/holder.log"; }
say "start on $SLURM_JOB_NODELIST (restart count ${SLURM_RESTART_COUNT:-0})"
for f in "$H"/running/*.sh; do [ -e "$f" ] && { say "requeue: back to queue $(basename "$f")"; mv "$f" "$H/queue/"; }; done
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    say "$n amdgpu $(srun --overlap -N1 -n1 -w "$n" cat /sys/module/amdgpu/version 2>/dev/null)"
done
echo "$SLURM_JOB_ID $SLURM_JOB_NODELIST" > "$H/ALLOCATION"
while [ ! -e "$H/STOP" ]; do
    item="$(ls "$H"/queue/*.sh 2>/dev/null | sort | head -1)"
    if [ -z "$item" ]; then sleep 30; continue; fi
    name="$(basename "$item" .sh)"
    mv "$item" "$H/running/" 2>/dev/null || continue
    say "run $name"
    bash "$H/running/$name.sh" > "$H/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$H/done/$name.rc"
    mv "$H/running/$name.sh" "$H/done/" 2>/dev/null
    say "done $name rc=$rc"
done
say "STOP found; exiting"
rm -f "$H/ALLOCATION"
