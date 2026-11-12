#!/usr/bin/env bash
# #48047 MI355X A/B, one point per job (plan B): build e1ab-sglang:1112 if this node lacks it, then the bench_serving
# generated-shared-prefix A/B (e1ab_item.sh gsp, ~22 min, inside the 30-min preemption exemption) on the 1node 4+4 shape
# for MIN_EVICT=$1. Requeued on preemption.
#   sbatch --job-name=llying-gsp-me<N> e1ab_gsp.sh <MIN_EVICT>
#SBATCH --partition=Compute-DCPT
#SBATCH --account=xiewen12
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -uo pipefail
me="${1:?MIN_EVICT}"
E=/perf_apps/liyingli/bench_agentx/e1ab
say() { echo "$(date -u +%FT%T) e1ab-gsp $SLURM_JOB_ID me=$me restart=${SLURM_RESTART_COUNT:-0} $*" | tee -a "$E/logs/e1ab.out"; }
say "start on $SLURM_JOB_NODELIST, src $(cat $E/src/REV), amdgpu $(cat /sys/module/amdgpu/version)"
# amdgpu 6.19.x crashes the engines on this stack (11-12); 6.16.13 is fine. Exclude 6.19 nodes at sbatch time too.
case "$(cat /sys/module/amdgpu/version)" in 6.19.*) say "amdgpu 6.19 node, not running"; exit 3 ;; esac
if ! docker image inspect e1ab-sglang:1112 >/dev/null 2>&1; then
    export DOCKER_CONFIG="$(mktemp -d /tmp/llying-docker.XXXX)"
    (cd "$E/image" && docker build --pull -t e1ab-sglang:1112 .) > "$E/logs/build-$SLURM_JOB_ID.log" 2>&1 ||
        { say "image build failed (logs/build-$SLURM_JOB_ID.log)"; exit 1; }
    say "image built: $(docker image inspect -f '{{.Id}}' e1ab-sglang:1112)"
fi
EST_MIN=25 MIN_EVICT=$me bash "$E/e1ab_item.sh" "gsp${GSP_TAG:-}-me$me" gsp 1node \
    > "$E/logs/gsp${GSP_TAG:-}-me$me-$SLURM_JOB_ID-r${SLURM_RESTART_COUNT:-0}.item.log" 2>&1
say "gsp exit $?"
