#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-gpuscript
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH -c 118
#SBATCH --mem 1200G
#SBATCH -t 00:20:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# Run one Python script (SCRIPT, relative to the work dir) in the engine image on one of the job's GPUs, with instr/ on
# PYTHONPATH and VLLM_UVA_VMM=1. Exits with the script's status.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" --export=ALL,SCRIPT=scripts/<name>.py[,NPROC=2] run_gpu_script.sbatch
# NPROC (1): >1 runs the script under torchrun on the first NPROC of the job's GPUs (host IPC and network for RCCL).
# SCRIPT_ENV: space-separated VAR=value pairs passed into the container.
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || exit 1
source scripts/env.sh
NPROC=${NPROC:-1}
GPU=$(echo "${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}" | cut -d, -f1-"$NPROC")
launch=(python3); ((NPROC > 1)) && launch=(python3 -m torch.distributed.run --nproc_per_node "$NPROC")
echo "== $(ts) job $SLURM_JOB_ID on $(hostname -s): ${SCRIPT:?SCRIPT=scripts/<name>.py} on GPU $GPU"
docker image inspect "$IMAGE" > /dev/null 2>&1 || timeout 1800 docker load -i "$IMAGE_TAR" > /dev/null 2>&1
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && gids+=(--group-add "$gid"); done
timeout 900 docker run --rm --ipc=host --network=host --device=/dev/kfd --device=/dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES="$GPU" \
    -e VLLM_UVA_VMM=1 -e PYTHONPATH="$W/instr" $(for kv in ${SCRIPT_ENV:-}; do printf -- '-e %s ' "$kv"; done) -v "$W/instr:$W/instr:ro" -v "$W/scripts:$W/scripts:ro" \
    --entrypoint "${launch[0]}" "$IMAGE" "${launch[@]:1}" "$W/$SCRIPT" 2>&1 | sed 's/^/   /'
rc=${PIPESTATUS[0]}
echo "== $(ts) $SCRIPT rc=$rc"
exit $rc
