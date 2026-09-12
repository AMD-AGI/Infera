#!/usr/bin/env bash
# Validate both nodes and the cross-node RDMA path before launching GLM-5.2.
# Usage: bash preflight.sh [mode|fabric|all]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

PROBE="$DIR/../../examples/sglang_1p1d_glm5.2/preflight_rdma.sh"
MODE="${1:-all}"
OUT="${PREFLIGHT_OUT_DIR:-$RUN_ROOT/preflight}"
mkdir -p "$OUT"
case "$MODE" in
    mode|fabric|all) ;;
    *) echo "usage: $0 [mode|fabric|all]" >&2; exit 64 ;;
esac

run_mode_probe() {
    local node="$1"
    echo "===== $node: registration mode ====="
    ssh $SSH_OPTS "$node" \
        "cd '$DIR/../..' && IMAGE='$IMAGE' \
        HOST_RDMA_LIB='$HOST_RDMA_LIB' HOST_RDMA_MOUNT='$HOST_RDMA_MOUNT' \
        bash '$PROBE' mode"
}

if [[ "$MODE" == "mode" || "$MODE" == "all" ]]; then
    run_mode_probe "$PREFILL_NODE" 2>&1 | tee "$OUT/mode_$PREFILL_NODE.log"
    run_mode_probe "$DECODE_NODE" 2>&1 | tee "$OUT/mode_$DECODE_NODE.log"
fi

if [[ "$MODE" == "fabric" || "$MODE" == "all" ]]; then
    attempt="${PREFLIGHT_ATTEMPT:-$(date -u +%Y%m%dT%H%M%SZ)}"
    dump="$OUT/fabric/$attempt"
    mkdir -p "$dump"
    echo "===== fabric: $PREFILL_NODE <-> $DECODE_NODE ====="
    echo "[preflight] output: $dump"

    docker_args="--rm --network host \
--device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
--group-add video --group-add render --cap-add=IPC_LOCK \
--ulimit memlock=-1:-1 --ulimit nofile='$NOFILE_ULIMIT' \
--security-opt seccomp=unconfined --ipc=host"
    if [[ -n "$HOST_RDMA_LIB" ]]; then
        docker_args+=" -v '$HOST_RDMA_LIB:$HOST_RDMA_MOUNT:ro'"
    fi
    probe_env="-e SLURM_PROCID -e SLURM_NNODES -e SLURMD_NODENAME \
-e PREFLIGHT_HOST -e PREFLIGHT_IMAGE \
-e INFERA_PREFLIGHT_RDMA_DEVICE='$RDMA_DEVICE' \
-e MOONCAKE_DISABLE_HIP_DMABUF='$MOONCAKE_DISABLE_HIP_DMABUF' \
-e MC_ENABLE_DEST_DEVICE_AFFINITY='$MC_ENABLE_DEST_DEVICE_AFFINITY' \
-e MC_TE_FILTERS='$MC_TE_FILTERS' \
-e MC_GID_INDEX='$MC_GID_INDEX' \
-e RDMAV_FORK_SAFE='$RDMAV_FORK_SAFE'"
    if [[ -n "${INFERA_PREFLIGHT_KV_GPUS:-}" ]]; then
        [[ "$INFERA_PREFLIGHT_KV_GPUS" =~ ^[0-9]+$ ]] || {
            echo "INFERA_PREFLIGHT_KV_GPUS must be a non-negative integer" >&2
            exit 64
        }
        probe_env+=" -e INFERA_PREFLIGHT_KV_GPUS='$INFERA_PREFLIGHT_KV_GPUS'"
    fi
    if [[ -n "${INFERA_PREFLIGHT_MOONCAKE_OPCODE:-}" ]]; then
        case "$INFERA_PREFLIGHT_MOONCAKE_OPCODE" in
            read|write) ;;
            *)
                echo "INFERA_PREFLIGHT_MOONCAKE_OPCODE must be read or write" >&2
                exit 64
                ;;
        esac
        probe_env+=" -e INFERA_PREFLIGHT_MOONCAKE_OPCODE='$INFERA_PREFLIGHT_MOONCAKE_OPCODE'"
    fi

    ssh $SSH_OPTS "$PREFILL_NODE" \
        "SLURM_PROCID=0 SLURM_NNODES=2 SLURMD_NODENAME='$PREFILL_NODE' \
        PREFLIGHT_HOST='$PREFILL_NODE' PREFLIGHT_IMAGE='$IMAGE' \
        docker run $docker_args $probe_env -v '$dump:$dump' '$IMAGE' \
        python3 -m infera.tools.preflight --dump-path '$dump' --netperf --mooncake" \
        >"$dump/$PREFILL_NODE.log" 2>&1 &
    p0=$!
    ssh $SSH_OPTS "$DECODE_NODE" \
        "SLURM_PROCID=1 SLURM_NNODES=2 SLURMD_NODENAME='$DECODE_NODE' \
        PREFLIGHT_HOST='$DECODE_NODE' PREFLIGHT_IMAGE='$IMAGE' \
        docker run $docker_args $probe_env -v '$dump:$dump' '$IMAGE' \
        python3 -m infera.tools.preflight --dump-path '$dump' --netperf --mooncake" \
        >"$dump/$DECODE_NODE.log" 2>&1 &
    p1=$!

    rc=0
    wait "$p0" || rc=1
    wait "$p1" || rc=1
    for log in "$dump/$PREFILL_NODE.log" "$dump/$DECODE_NODE.log"; do
        echo "----- $log -----"
        awk '{print}' "$log"
    done
    [[ "$rc" == "0" ]] || { echo "[preflight] fabric probe failed" >&2; exit 1; }
fi
