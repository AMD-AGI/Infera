#!/usr/bin/env bash
#SBATCH -J lyl-ix5138-reap
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -c 1
#SBATCH --mem 1G
#SBATCH -t 00:10:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/ix5138-crusoe-20261120/logs/reap-%j.out
# Safety net after campaign jobs: a job killed at its time limit leaves its docker containers running on the node, so
# this removes every lyl-ix5138-* container on the given nodes. Submit with -N <n> -w <nodes>
# --dependency=afterany:<jobs>. Every step exits 0.
srun -D /tmp -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" /bin/bash -c \
    'ids=$(docker ps -aq --filter name=lyl-ix5138-); echo "$(hostname -s): removing $(wc -w <<<"$ids") container(s)";
     [ -n "$ids" ] && docker rm -f $ids >/dev/null; exit 0' </dev/null
exit 0
