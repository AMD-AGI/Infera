#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --time=10:00:00
#SBATCH --job-name=llying-adaptive-c80
set -Eeuo pipefail
printf 'ALLOCATION_READY job=%s nodes=%s utc=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$(date -u --iso-8601=seconds)"
trap 'exit 0' TERM INT
sleep 36000 &
wait $!
