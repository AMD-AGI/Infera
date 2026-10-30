#!/usr/bin/env bash
# Two-node holder for the infera-sglang AUS e2e test. Holds the nodes with the resources srtctl's
# own sbatch asks for, and runs queued work items inside this allocation, one at a time, oldest
# first. A work item is a bash script (typically the sbatch_script.sh srtctl generated for one
# matrix point, so do_sweep and everything below it run exactly as in CI's job).
#   queue:   $H/queue/<NN>-<label>.sh       (drop a script here to run it)
#   running: $H/running/                    (moved back to queue/ if the job is requeued)
#   done:    $H/done/<name>  + $H/logs/<name>.log, exit code in $H/done/<name>.rc
#   stop:    touch $H/STOP                  (exit after the current item)
#SBATCH --job-name=llying-holder
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
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/apps/liyingli/holder/holder-%j.out
set -uo pipefail
H=/apps/liyingli/holder
mkdir -p "$H"/{queue,running,done,logs}
say() { echo "$(date -u +%FT%TZ) holder $SLURM_JOB_ID: $*"; }
say "start on $SLURM_JOB_NODELIST (restart count ${SLURM_RESTART_COUNT:-0})"
# A requeue kills the item that was running; put it back at the head of the queue.
for f in "$H"/running/*.sh; do [ -e "$f" ] && { say "requeue: back to queue $(basename "$f")"; mv "$f" "$H/queue/"; }; done
echo "$SLURM_JOB_ID $SLURM_JOB_NODELIST" > "$H/ALLOCATION"
while [ ! -e "$H/STOP" ]; do
    item="$(ls "$H"/queue/*.sh 2>/dev/null | sort | head -1)"
    if [ -z "$item" ]; then sleep 30; continue; fi
    name="$(basename "$item" .sh)"
    mv "$item" "$H/running/"
    say "run $name"
    bash "$H/running/$name.sh" > "$H/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$H/done/$name.rc"
    mv "$H/running/$name.sh" "$H/done/"
    say "done $name rc=$rc"
done
say "STOP found; exiting"
rm -f "$H/ALLOCATION"
