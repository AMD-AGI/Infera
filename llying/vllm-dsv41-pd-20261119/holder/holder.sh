#!/usr/bin/env bash
# One-node holder for the DSv4.1-Flash infera+vLLM bring-up. Runs queued work items inside this allocation,
# one at a time, oldest first (same layout as /apps/liyingli/jobs/holder.sh). Several holder jobs may share the queue
# (a short backfill job and a long one): an item records its owner job in running/<name>.job.
#   queue:   $H/queue/<NN>-<label>.sh       (drop a script here to run it)
#   running: $H/running/                    (moved back to queue/ if its owner job is gone)
#   done:    $H/done/<name>  + $H/logs/<name>.log, exit code in $H/done/<name>.rc
#   stop:    touch $H/STOP                  (exit after the current item)
#SBATCH --job-name=llying-dsv41-ivllm
#SBATCH --partition=Compute-DCPT
#SBATCH --account=emad
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder/holder-%j.out
set -uo pipefail
H=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder
mkdir -p "$H"/{queue,running,done,logs}
say() { echo "$(date -u +%FT%TZ) holder $SLURM_JOB_ID: $*"; }
say "start on $SLURM_JOB_NODELIST (restart count ${SLURM_RESTART_COUNT:-0})"
for f in "$H"/running/*.sh; do
    [ -e "$f" ] || continue
    o="$(cat "${f%.sh}.job" 2>/dev/null)"
    if [ -n "$o" ] && [ "$o" != "$SLURM_JOB_ID" ] && [ "$(squeue -h -j "$o" -o %T 2>/dev/null)" = RUNNING ]; then
        say "running under job $o, left alone: $(basename "$f")"; continue
    fi
    say "requeue: back to queue $(basename "$f")"; rm -f "${f%.sh}.job"; mv "$f" "$H/queue/"
done
echo "$SLURM_JOB_ID $SLURM_JOB_NODELIST" >> "$H/ALLOCATION"
while [ ! -e "$H/STOP" ]; do
    item="$(ls "$H"/queue/*.sh 2>/dev/null | sort | head -1)"
    if [ -z "$item" ]; then sleep 30; continue; fi
    # Preemption ends the extern step ~5 min before the kill; do not start an item in that window.
    x="$(sacct -n -j "$SLURM_JOB_ID.extern" -o State 2>/dev/null | head -1 | tr -d ' ')"
    case "$x" in COMPLETED* | CANCELLED* | FAILED*) say "extern step $x (preemption notice?): no new items"; exec sleep 100000 ;; esac
    name="$(basename "$item" .sh)"
    mv "$item" "$H/running/" 2>/dev/null || continue
    echo "$SLURM_JOB_ID" > "$H/running/$name.job"
    say "run $name"
    ITEM_NAME="$name" bash "$H/running/$name.sh" > "$H/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$H/done/$name.rc"
    mv "$H/running/$name.sh" "$H/done/" 2>/dev/null; rm -f "$H/running/$name.job"
    say "done $name rc=$rc"
done
say "STOP found; exiting"
