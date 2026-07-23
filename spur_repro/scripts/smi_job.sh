#!/bin/bash
#SBATCH --job-name=smi_test
#SBATCH --account=amd-primus
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=00:05:00

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -u) ==="
echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
echo "--- amd-smi list ---"
amd-smi list 2>&1 | head -40 || echo "amd-smi not found"
echo "--- rocm-smi --showproductname ---"
rocm-smi --showproductname 2>&1 | head -20 || echo "rocm-smi not found"
echo "=== done at $(date -u) ==="
