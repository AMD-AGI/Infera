#!/usr/bin/env bash
# DRIVER619-20261113: two-node holder for the amdgpu 6.19 KFD accounting work, one amdgpu 6.19 node and one 6.16 node
# (submit with -w <a 6.19 node> and --exclude=<the other 6.19 nodes>). Records each node's driver, then runs queued
# items one at a time, oldest first; an item is a bash script run on the batch host with JOB, NODE619, NODE616 and K set
# (run on a node with: srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 ...).
#   queue: $K/queue/<NN>-<label>.sh   running: $K/running/   done: $K/done/<name>.sh + .rc   logs: $K/logs/<name>.log
#   stop:  touch $K/STOP (exit after the current item)
#SBATCH --job-name=llying-kfd619
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
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -uo pipefail
K=/perf_apps/liyingli/bench_agentx/kfd619
O=/perf_apps/liyingli/bench_agentx/opt-aus-20261103
mkdir -p "$K"/{queue,running,done,logs}
say() { echo "$(date -u +%FT%TZ) kfd619 $SLURM_JOB_ID: $*" | tee -a "$K/logs/holder.log"; }
say "start on $SLURM_JOB_NODELIST (restart count ${SLURM_RESTART_COUNT:-0})"
for f in "$K"/running/*.sh; do [ -e "$f" ] && { say "requeue: back to queue $(basename "$f")"; mv "$f" "$K/queue/"; }; done
NODE619="" NODE616=""
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    v="$(srun --overlap -N1 -n1 -w "$n" cat /sys/module/amdgpu/version 2>/dev/null)"
    say "$n amdgpu $v"
    if [[ "$v" == 6.19.* ]]; then [ -z "$NODE619" ] && NODE619=$n || NODE616=${NODE616:-$n}; else [ -z "$NODE616" ] && NODE616=$n || NODE619=${NODE619:-$n}; fi
    r="$(srun --overlap -N1 -n1 -w "$n" bash "$O/scripts/clear_foreign_containers.sh" "$SLURM_JOB_ID" 2>&1)"
    [ -z "$r" ] || say "foreign containers: $r"
done
export JOB=$SLURM_JOB_ID NODE619 NODE616 K
echo "$JOB NODE619=$NODE619 NODE616=$NODE616" > "$K/ALLOCATION"
say "NODE619=$NODE619 NODE616=$NODE616"
while [ ! -e "$K/STOP" ]; do
    item="$(ls "$K"/queue/*.sh 2>/dev/null | sort | head -1)"
    if [ -z "$item" ]; then sleep 30; continue; fi
    x="$(sacct -n -j "$SLURM_JOB_ID.extern" -o State 2>/dev/null | head -1 | tr -d ' ')"
    case "$x" in COMPLETED* | CANCELLED* | FAILED*) say "extern step $x (preemption notice?): no new items"; exec sleep 100000 ;; esac
    name="$(basename "$item" .sh)"
    mv "$item" "$K/running/" 2>/dev/null || continue
    say "run $name"
    bash "$K/running/$name.sh" > "$K/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$K/done/$name.rc"
    mv "$K/running/$name.sh" "$K/done/" 2>/dev/null
    say "done $name rc=$rc"
done
say "STOP found; exiting"
rm -f "$K/ALLOCATION"
