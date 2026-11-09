#!/usr/bin/env bash
# One-node holder (llying-holder1): single-node items (image builds, one-node 4+4 gates) while the two-node holder waits.
# own sbatch asks for, and runs queued work items inside this allocation, one at a time, oldest
# first. A work item is a bash script (typically the sbatch_script.sh srtctl generated for one
# matrix point, so do_sweep and everything below it run exactly as in CI's job).
#   queue:   $H/queue/<NN>-<label>.sh       (drop a script here to run it)
#   running: $H/running/                    (moved back to queue/ if the job is requeued)
#   done:    $H/done/<name>  + $H/logs/<name>.log, exit code in $H/done/<name>.rc
#   stop:    touch $H/STOP                  (exit after the current item)
#SBATCH --job-name=llying-holder1
#SBATCH --partition=Compute-DCPT
#SBATCH --account=xiewen12
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/apps/liyingli/holder1/holder1-%j.out
set -uo pipefail
H=/apps/liyingli/holder1
mkdir -p "$H"/{queue,running,done,logs}
say() { echo "$(date -u +%FT%TZ) holder $SLURM_JOB_ID: $*"; }
say "start on $SLURM_JOB_NODELIST (restart count ${SLURM_RESTART_COUNT:-0})"
# A requeue kills the item that was running; put it back at the head of the queue. Holders can share one queue
# (several one-node jobs at once): an item whose owner job (running/<name>.job) is still running is left alone.
for f in "$H"/running/*.sh; do
    [ -e "$f" ] || continue
    o="$(cat "${f%.sh}.job" 2>/dev/null)"
    if [ -n "$o" ] && [ "$o" != "$SLURM_JOB_ID" ] && [ "$(squeue -h -j "$o" -o %T 2>/dev/null)" = RUNNING ]; then
        say "running under job $o, left alone: $(basename "$f")"; continue
    fi
    say "requeue: back to queue $(basename "$f")"; rm -f "${f%.sh}.job"; mv "$f" "$H/queue/"
done
echo "$SLURM_JOB_ID $SLURM_JOB_NODELIST" > "$H/ALLOCATION"
while [ ! -e "$H/STOP" ]; do
    item="$(ls "$H"/queue/*.sh 2>/dev/null | sort | head -1)"
    if [ -z "$item" ]; then sleep 30; continue; fi
    # Preemption ends the extern step (ssh is refused from then on) and signals the steps ~5 min before the kill; an item
    # started in that window would leave engines its guard can no longer remove (11-07 05:17 n02-29, 11-09 10:35 n02-33).
    x="$(sacct -n -j "$SLURM_JOB_ID.extern" -o State 2>/dev/null | head -1 | tr -d ' ')"
    case "$x" in COMPLETED* | CANCELLED* | FAILED*) say "extern step $x (preemption notice?): no new items"; exec sleep 100000 ;; esac
    name="$(basename "$item" .sh)"
    mv "$item" "$H/running/" 2>/dev/null || continue
    echo "$SLURM_JOB_ID" > "$H/running/$name.job"
    say "run $name"
    bash "$H/running/$name.sh" > "$H/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$H/done/$name.rc"
    # An item that deferred went back to the queue and may already run under another holder: leave its entry alone.
    if [ "$(cat "$H/running/$name.job" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then
        mv "$H/running/$name.sh" "$H/done/" 2>/dev/null; rm -f "$H/running/$name.job"
    fi
    say "done $name rc=$rc"
done
say "STOP found; exiting"
rm -f "$H/ALLOCATION"
