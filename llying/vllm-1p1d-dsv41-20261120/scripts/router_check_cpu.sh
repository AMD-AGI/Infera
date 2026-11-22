#!/usr/bin/env bash
# CPU-only Slurm job: cargo fmt / clippy -D warnings / test for the router, same gate as CI's `rust` job.
# Small and short so backfill can slot it into a gap; no GPU, no image beyond rust:1-bookworm.
#SBATCH --job-name=llying-router-cpu
#SBATCH --partition=Compute-DCPT
#SBATCH --account=emad
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/logs/router-cpu-%j.out
set -uo pipefail
source /perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/scripts/env.sh
REPO=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
IMG=docker.io/library/rust:1-bookworm

echo "$(ts) router-cpu: $(hostname -s) job ${SLURM_JOB_ID:-?}, head $(git -C "$REPO" rev-parse --short HEAD)"
pull_image "$IMG" > /dev/null || { echo "$(ts) pull of $IMG FAILED"; exit 1; }
docker run --rm --cpus 32 --memory 60g -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$REPO/rust:/work/rust" -v "$W/scripts/router_check_inner.sh:/inner.sh:ro" "$IMG" bash /inner.sh
rc=$?
echo "$(ts) router-cpu: checks rc=$rc"
git -C "$REPO" diff --stat -- rust
exit $rc
