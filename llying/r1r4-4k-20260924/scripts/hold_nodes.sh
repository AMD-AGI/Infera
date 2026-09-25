#!/usr/bin/env bash
#SBATCH --job-name=llying-r1r4-c80
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --account=emad
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --exclude=smci355-ccs-aus-n04-29
#SBATCH --output=/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/events/allocation-%j.log
set -eu
printf 'job=%s nodes=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
sleep 86400
