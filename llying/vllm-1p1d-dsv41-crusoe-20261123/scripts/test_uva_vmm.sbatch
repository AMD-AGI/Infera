#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-uvatest
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
# GPU test of instr/uva_vmm.py (vLLM UvaBuffer in HIP VMM host memory): scripts/uva_vmm_test.py in the engine image on
# one of the job's GPUs. Fails (non-zero exit) when the test does, so a run that uses the overlay can depend on it.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" test_uva_vmm.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || exit 1
source scripts/env.sh
GPU=${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}; GPU=${GPU%%,*}
echo "== $(ts) job $SLURM_JOB_ID on $(hostname -s): uva_vmm test on GPU $GPU"
docker image inspect "$IMAGE" > /dev/null 2>&1 || timeout 1800 docker load -i "$IMAGE_TAR" > /dev/null 2>&1
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && gids+=(--group-add "$gid"); done
timeout 900 docker run --rm --device=/dev/kfd --device=/dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES="$GPU" \
    -e VLLM_UVA_VMM=1 -e PYTHONPATH="$W/instr" -v "$W/instr:$W/instr:ro" -v "$W/scripts:$W/scripts:ro" \
    --entrypoint python3 "$IMAGE" "$W/scripts/uva_vmm_test.py" 2>&1 | sed 's/^/   /'
rc=${PIPESTATUS[0]}
echo "== $(ts) uva_vmm test rc=$rc"
exit $rc
