#!/usr/bin/env bash
# rail_probe (task g4tune xnode, 2026-10-21): Mooncake RDMA preflight between two nodes with BOTH ends pinned to
# the same ionic rail (copy of the 09-18 samerail pin_probe.sh, v0521 image, 135/136, output on NFS under W).
# Usage (login host or a node with ssh to both): rail_probe.yihou.sh <ionic_dev> <nodeA> <nodeB>
# Artifacts: $W/xnode/preflight/<ionic_dev>-<UTC>/rank-*.log + preflight dumps. Containers: --rm (our own).
set -euo pipefail
DEV="${1:?ionic dev}"; A="${2:?node A}"; B="${3:?node B}"
[[ "$DEV" =~ ^ionic_[0-7]$ ]] || { echo "bad dev $DEV" >&2; exit 2; }
IMAGE="${IMAGE:-infera-sglang:v0521-yihou-1010-mc4340}"
HOST_RDMA_LIB="${HOST_RDMA_LIB:-/lib/x86_64-linux-gnu/libionic.so}"
W=/shared_nfs/yihou_glm52/g4tune_20261019
OUT="$W/xnode/preflight/$DEV-$(date -u +%Y%m%dT%H%M%SZ)"
[[ ! -e "$OUT" ]] || { echo "exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
TOKEN="yihou-rail-$DEV-$(date -u +%Y%m%dT%H%M%S)-$$"
NODES=("$A" "$B")
run_rank(){
    local node="$1"; local rank="$2"; local name="yihou-railprobe-$TOKEN-rank$rank"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" timeout 900 \
        docker run --rm --network host --ipc host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add IPC_LOCK \
        --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        --security-opt seccomp=unconfined \
        -v "$HOST_RDMA_LIB:/host-libionic/libionic.so:ro" \
        --name "$name" \
        -e "SLURM_PROCID=$rank" -e "SLURM_NNODES=2" \
        -e "SLURMD_NODENAME=$node" -e "PREFLIGHT_HOST=$node" \
        -e "PREFLIGHT_IMAGE=$IMAGE" \
        -e "INFERA_PREFLIGHT_RUN_ID=$TOKEN" \
        -e "INFERA_PREFLIGHT_MOONCAKE_OPCODE=write" \
        -e "INFERA_PREFLIGHT_RDMA_DEVICE=$DEV" \
        -e "MC_GID_INDEX=1" \
        -v "$OUT:$OUT" \
        "$IMAGE" python3 -m infera.tools.preflight \
        --dump-path "$OUT" --mooncake \
        > "$OUT/rank-$rank-$node.log" 2>&1
}
pids=()
for i in 0 1; do run_rank "${NODES[i]}" "$i" & pids+=($!); done
st=0; for p in "${pids[@]}"; do wait "$p" || st=1; done
echo "RAIL_PROBE dev=$DEV exit=$st out=$OUT"
exit "$st"
