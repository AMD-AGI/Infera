#!/usr/bin/env bash
# Two-node allocation for the GLM-5.2 P4D4 work: one queue runner (holder.sh) per node. Chain the next allocation with
#   sbatch -d afterany:<previous job> holder.sbatch
#SBATCH --job-name=llying-p4d4-aus
#SBATCH --partition=Compute-DCPT
#SBATCH --account=emad
#SBATCH --qos=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/logs/holder-%j.out
srun --ntasks-per-node=1 --kill-on-bad-exit=0 bash /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/holder/holder.sh
