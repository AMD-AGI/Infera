#!/usr/bin/env bash
# Byte-check the production Mooncake WRITE path for every Prefill→Decode pair.
set -euo pipefail
COMPONENT=preflight
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
init_ssh
start_log
OUT_DIR="${OUT_DIR:-${RUN_LOG%.log}.d}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
    die "output directory is not empty: $OUT_DIR"
fi
mkdir -p "$OUT_DIR/mode" "$OUT_DIR/write"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

gpu_count() { awk -F, '{print NF}' <<<"$1"; }
prefill_gpu_count="$(gpu_count "$PREFILL_GPU_DEVICES")"
decode_gpu_count="$(gpu_count "$DECODE_GPU_DEVICES")"
[[ "$prefill_gpu_count" == "$decode_gpu_count" ]] ||
    die "preflight requires equal P/D GPU counts; got $prefill_gpu_count and $decode_gpu_count"
expected_gpus="$prefill_gpu_count"

docker_base=(
    docker run --rm --network host --ipc host
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render --cap-add IPC_LOCK
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    --security-opt seccomp=unconfined
)
if [[ -n "${HOST_RDMA_LIB:-}" ]]; then
    docker_base+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")
fi

run_token="$(date -u +%Y%m%dT%H%M%S)-$$"
declare -a active_nodes=() active_names=()
cleanup_active() {
    local index
    set +e
    for index in "${!active_nodes[@]}"; do
        ssh_exec "${active_nodes[index]}" docker stop --time 30 "${active_names[index]}" \
            >/dev/null 2>&1
        ssh_exec "${active_nodes[index]}" docker rm -f "${active_names[index]}" \
            >/dev/null 2>&1
    done
    set -e
}
trap cleanup_active EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    log "registration mode: $instance on $node"
    mode_name="$CONTAINER_PREFIX-preflight-mode-$instance-$run_token"
    active_nodes=("$node")
    active_names=("$mode_name")
    ssh_exec "$node" timeout "${PREFLIGHT_TIMEOUT:-1800}" \
        "${docker_base[@]}" \
        --name "$mode_name" \
        -e "HIP_VISIBLE_DEVICES=$gpus" \
        -e "INFERA_PREFLIGHT_RDMA_DEVICE=$RDMA_DEVICE" \
        -e "INFERA_PREFLIGHT_GID_INDEX=$MC_GID_INDEX" \
        -e "MC_GID_INDEX=$MC_GID_INDEX" \
        -e "MC_TE_FILTERS=$MC_TE_FILTERS" \
        -e "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE" \
        "$IMAGE" python3 -m infera.tools.preflight.mooncake_mode \
        >"$OUT_DIR/mode/$instance.log" 2>&1 </dev/null
    cleanup_active
    active_nodes=()
    active_names=()
done < <(topology_rows)

run_rank() {
    local node="$1" gpus="$2" rank="$3" host="$4" pair="$5" name="$6"
    ssh_exec "$node" timeout "${PREFLIGHT_TIMEOUT:-1800}" \
        "${docker_base[@]}" \
        --name "$name" \
        -e "HIP_VISIBLE_DEVICES=$gpus" \
        -e "SLURM_PROCID=$rank" -e SLURM_NNODES=2 \
        -e "SLURMD_NODENAME=$host" -e "PREFLIGHT_HOST=$host" \
        -e "PREFLIGHT_IMAGE=$IMAGE" \
        -e INFERA_PREFLIGHT_MOONCAKE_OPCODE=write \
        -e "INFERA_PREFLIGHT_KV_GPUS=$expected_gpus" \
        -e "INFERA_PREFLIGHT_RDMA_DEVICE=$RDMA_DEVICE" \
        -e "INFERA_PREFLIGHT_GID_INDEX=$MC_GID_INDEX" \
        -e "MC_GID_INDEX=$MC_GID_INDEX" \
        -e "MC_TE_FILTERS=$MC_TE_FILTERS" \
        -e MC_DISABLE_HIP_TRANSPORT=1 \
        -e "MC_ENABLE_DEST_DEVICE_AFFINITY=$MC_ENABLE_DEST_DEVICE_AFFINITY" \
        -e "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF" \
        -e "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE" \
        -v "$pair:$pair" \
        "$IMAGE" python3 -m infera.tools.preflight \
        --dump-path "$pair" --mooncake --collect-only
}

declare -a p_instances=() p_nodes=() p_gpus=()
declare -a d_instances=() d_nodes=() d_gpus=()
while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    if [[ "$role" == prefill ]]; then
        p_instances+=("$instance"); p_nodes+=("$node"); p_gpus+=("$gpus")
    else
        d_instances+=("$instance"); d_nodes+=("$node"); d_gpus+=("$gpus")
    fi
done < <(topology_rows)

for ((p = 0; p < ${#p_nodes[@]}; p++)); do
    for ((d = 0; d < ${#d_nodes[@]}; d++)); do
        pair="$OUT_DIR/write/${p_instances[p]}__to__${d_instances[d]}"
        mkdir -p "$pair"
        python3 - "$pair/pair.json" "${p_instances[p]}" "${p_nodes[p]}" \
            "${d_instances[d]}" "${d_nodes[d]}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "source_instance": sys.argv[2],
    "source_node": sys.argv[3],
    "destination_instance": sys.argv[4],
    "destination_node": sys.argv[5],
    "operation": "write",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
        log "WRITE ${p_instances[p]}(${p_nodes[p]}) -> ${d_instances[d]}(${d_nodes[d]})"
        rank0_name="$CONTAINER_PREFIX-preflight-${p_instances[p]}-${d_instances[d]}-r0-$run_token"
        rank1_name="$CONTAINER_PREFIX-preflight-${p_instances[p]}-${d_instances[d]}-r1-$run_token"
        active_nodes=("${p_nodes[p]}" "${d_nodes[d]}")
        active_names=("$rank0_name" "$rank1_name")
        run_rank "${p_nodes[p]}" "${p_gpus[p]}" 0 "${p_nodes[p]}" "$pair" "$rank0_name" \
            >"$pair/rank0.log" 2>&1 &
        rank0=$!
        run_rank "${d_nodes[d]}" "${d_gpus[d]}" 1 "${d_nodes[d]}" "$pair" "$rank1_name" \
            >"$pair/rank1.log" 2>&1 &
        rank1=$!
        pair_rc=0
        wait "$rank0" || pair_rc=1
        wait "$rank1" || pair_rc=1
        cleanup_active
        active_nodes=()
        active_names=()
        (( pair_rc == 0 )) || log "pair process failed; validator will report details"
    done
done

python3 "$DIR/tools/validate_preflight.py" \
    --topology "$TOPOLOGY_FILE" \
    --root "$OUT_DIR/write" \
    --expected-gpus "$expected_gpus" \
    --output "$OUT_DIR/validation.json"
log "PASS: artifacts=$OUT_DIR"
