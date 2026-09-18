#!/usr/bin/env bash
# Purpose: decide whether Mooncake's per-GPU VRAM failure on rails 4-7 is caused
#   by the two ends auto-selecting DIFFERENT NICs (cross-rail, unreachable) or by
#   the rails themselves being broken.
# Method: mirror preflight.sh's docker run exactly, but add
#   INFERA_PREFLIGHT_RDMA_DEVICE so BOTH ends are pinned to the SAME NIC. Rails
#   are physically isolated (ionic_i <-> ionic_i only), so if pinning makes the
#   GPU variants pass, auto-selection divergence was the cause.
# Usage: ./pin_probe.sh <ionic_dev> <out_subdir>
# Artifacts: results/yihou-1p1d-c64/preflight/<out_subdir>/
set -euo pipefail

DEV="${1:?usage: pin_probe.sh <ionic_dev> <out_subdir>}"
SUB="${2:?usage: pin_probe.sh <ionic_dev> <out_subdir>}"

IMAGE="${IMAGE:-infera-sglang:v0519-yihou-0917}"
HOST_RDMA_LIB="${HOST_RDMA_LIB:-/lib/x86_64-linux-gnu/libionic.so}"
NODES=(crsuse2-m2m-135 crsuse2-m2m-138)
BASE="/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-1p1d-c64/preflight"
OUT="$BASE/$SUB"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no)

[[ ! -e "$OUT" ]] || { echo "output path already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

TOKEN="pin-$DEV-$(date -u +%Y%m%dT%H%M%S)-$$"

run_rank() {
    # Separate statements: bash expands every word of a `local` line before it
    # assigns any of them, so `name` cannot reference `rank` on the same line.
    local node="$1"
    local rank="$2"
    local name="yihou-pinprobe-$TOKEN-rank$rank"
    ssh -n "${SSH_OPTS[@]}" "$node" timeout 900 \
        docker run --rm --network host --ipc host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add IPC_LOCK \
        --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        --security-opt seccomp=unconfined \
        -v "$HOST_RDMA_LIB:/host-libionic/libionic.so:ro" \
        --name "$name" \
        -e "SLURM_PROCID=$rank" -e "SLURM_NNODES=${#NODES[@]}" \
        -e "SLURMD_NODENAME=$node" -e "PREFLIGHT_HOST=$node" \
        -e "PREFLIGHT_IMAGE=$IMAGE" \
        -e "INFERA_PREFLIGHT_RUN_ID=$TOKEN" \
        -e "INFERA_PREFLIGHT_MOONCAKE_OPCODE=write" \
        -e "INFERA_PREFLIGHT_RDMA_DEVICE=$DEV" \
        -e "MC_GID_INDEX=1" \
        -v "$OUT:$OUT" \
        "$IMAGE" python3 -m infera.tools.preflight \
        --dump-path "$OUT" --mooncake \
        >"$OUT/rank-$rank-$node.log" 2>&1
}

pids=()
for i in "${!NODES[@]}"; do
    run_rank "${NODES[i]}" "$i" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "PIN_PROBE dev=$DEV exit=$status out=$OUT"
exit "$status"
