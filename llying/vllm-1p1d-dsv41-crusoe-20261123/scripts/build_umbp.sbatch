#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-buildumbp
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH -c 118
#SBATCH --mem 1200G
#SBATCH -t 02:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/job-%j.out
# Build MORI's umbp_standalone_server with our patch (instr/mori-v1.2.3.post1-umbp-pinned-dram.patch, applied in SRC)
# inside the engine image, so it links the image's ROCm and gRPC; the binary then replaces the image's through a bind
# mount. CPU only. Output in logs/build-umbp-<job>/.
#   sbatch -x "$(paste -sd, exclude_nodes.txt)" [--export=ALL,SRC=<mori tree>] build_umbp.sbatch
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
cd "$W" || exit 0
source scripts/env.sh
SRC=${SRC:-/home/liyingli/bench_agentx/src/mori-v1.2.3.post1}
OUT=$W/logs/build-umbp-$SLURM_JOB_ID
mkdir -p "$OUT" && chmod 777 "$OUT"
echo "== $(ts) job $SLURM_JOB_ID on $(hostname -s): building umbp_standalone_server from $SRC ($(git -C "$SRC" log --oneline -1))"
docker image inspect "$IMAGE" > /dev/null 2>&1 || timeout 1800 docker load -i "$IMAGE_TAR" > /dev/null 2>&1
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "== no $IMAGE on this node"; exit 0; }
git -C "$SRC" diff v1.2.3.post1 --stat > "$OUT/patch-stat.txt" 2>&1
timeout 5400 docker run --rm --network host -v "$SRC:$SRC:ro" -v "$OUT:$OUT" -v "$W/scripts:$W/scripts:ro" \
    --entrypoint bash "$IMAGE" "$W/scripts/build_umbp_inner.sh" "$SRC" "$OUT" > "$OUT/build.log" 2>&1
echo "== $(ts) build rc=$?: $(tail -1 "$OUT/build.log")"
exit 0
