#!/usr/bin/env bash
# CPU-only Slurm job: bump the router's hyper past hyperium/hyper#4143 and run the CI gate; also report what
# origin/main pins and which commit brought hyper 1.10 in.
#SBATCH --job-name=llying-router-cpu
#SBATCH --partition=Compute-DCPT
#SBATCH --account=emad
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/logs/router-hyper-%j.out
set -uo pipefail
source /perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/scripts/env.sh
REPO=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
IMG=docker.io/library/rust:1-bookworm
HYPER_TO=${HYPER_TO:-1.12.0}
# Submitted to more than one partition; whichever starts first does the work.
mkdir "$W/logs/router-hyper.lock" 2>/dev/null || { echo "$(ts) another copy already ran; exiting"; exit 0; }

echo "$(ts) router-hyper: $(hostname -s) job ${SLURM_JOB_ID:-?}, head $(git -C "$REPO" rev-parse --short HEAD), target hyper $HYPER_TO"
git -C "$REPO" fetch -q origin main 2>&1 | tail -2
echo "origin/main ($(git -C "$REPO" rev-parse --short origin/main)) pins hyper $(git -C "$REPO" show origin/main:rust/Cargo.lock | grep -A1 '^name = "hyper"$' | tail -1)"
echo "hyper version history in rust/Cargo.lock (newest first, changes only):"
prev=""
for c in $(git -C "$REPO" log --format=%h -n 80 origin/main -- rust/Cargo.lock); do
    v=$(git -C "$REPO" show "$c:rust/Cargo.lock" 2>/dev/null | grep -A1 '^name = "hyper"$' | tail -1)
    [ "$v" != "$prev" ] && echo "  $(git -C "$REPO" log -1 --format='%h %ad %s' --date=short "$c") -> $v"
    prev=$v
done | head -6

pull_image "$IMG" > /dev/null || { echo "$(ts) pull of $IMG FAILED"; exit 1; }
docker run --rm --cpus "${CPUS:-32}" --cpu-shares "${CPU_SHARES:-1024}" --memory 60g -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e HYPER_TO="$HYPER_TO" \
    -v "$REPO/rust:/work/rust" -v "$W/scripts/router_bump_inner.sh:/inner.sh:ro" "$IMG" bash /inner.sh
rc=$?
echo "$(ts) router-hyper: checks rc=$rc"
git -C "$REPO" diff --stat -- rust
exit $rc
