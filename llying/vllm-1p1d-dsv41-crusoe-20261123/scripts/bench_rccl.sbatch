#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-rccl
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:2
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH -t 00:40:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# Two-GPU RCCL micro-benchmark at the prefill leg's MoE message size (scripts/rccl_bench.py) under several
# NCCL_* settings, in the engine image. Run-and-exit; always exits 0.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" bench_rccl.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || exit 0
source scripts/env.sh
ts() { date -u +%FT%TZ; }
echo "== $(ts) job $SLURM_JOB_ID on $(hostname -s): ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "-- loading $IMAGE_TAR"; docker load -i "$IMAGE_TAR" > /dev/null 2>&1; }
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "== image unavailable; ending"; exit 0; }
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
OUT=$W/logs/rccl-bench-$SLURM_JOB_ID.txt
timeout 1500 docker run --rm --name "lyl-dsv41-rccl-$SLURM_JOB_ID" --network host --ipc host --shm-size 16g \
    --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}" \
    -v "$W/scripts:/bench:ro" --entrypoint bash "$IMAGE" -c '
        for cfg in "" "NCCL_MIN_NCHANNELS=16" "NCCL_MIN_NCHANNELS=32" "NCCL_MIN_NCHANNELS=64" "NCCL_PROTO=Simple" \
                   "NCCL_MIN_NCHANNELS=32 NCCL_PROTO=Simple" "NCCL_MIN_NCHANNELS=64 NCCL_PROTO=Simple"; do
            env $cfg timeout 300 python3 /bench/rccl_bench.py "${cfg:-default}" 2>&1 | grep -E "GB/s|Error|error" | head -6
        done' > "$OUT" 2>&1
echo "== $(ts) done: $(wc -l < "$OUT") lines in $OUT"
docker rm -f "lyl-dsv41-rccl-$SLURM_JOB_ID" > /dev/null 2>&1
exit 0
