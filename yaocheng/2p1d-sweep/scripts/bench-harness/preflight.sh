#!/usr/bin/env bash
# Purpose: Run Infera's multi-node network and Mooncake preflight once.
# Usage: IMAGE=<image> ./preflight.sh NODE [NODE ...]
#   Comma-separated input is also accepted: IMAGE=<image> ./preflight.sh node-a,node-b
# Artifacts: rank logs, raw per-node JSON, Mooncake intermediates, and HTML report.
# Artifact paths: OUT_DIR, default .tmp/results/<UTC>-preflight.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_TMP_DIR="${SWEEP_TMP_DIR:-$(cd "$DIR/../.." && pwd)/.tmp}"

: "${IMAGE:?usage: IMAGE=<image> $0 NODE [NODE ...]}"
(( $# > 0 )) || { echo "usage: IMAGE=<image> $0 NODE [NODE ...]" >&2; exit 2; }

nodes=()
declare -A seen=()
for argument in "$@"; do
    IFS=',' read -r -a items <<<"$argument"
    for node in "${items[@]}"; do
        [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] || {
            echo "invalid node name: $node" >&2
            exit 2
        }
        if [[ -z "${seen[$node]:-}" ]]; then
            nodes+=("$node")
            seen["$node"]=1
        fi
    done
done
(( ${#nodes[@]} >= 2 )) || {
    echo "preflight requires at least two distinct nodes" >&2
    exit 2
}

read -r -a ssh_args <<<"${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
remote_command() {
    local result="" item quoted
    for item in "$@"; do
        printf -v quoted '%q' "$item"
        result+="${result:+ }$quoted"
    done
    echo "$result"
}
ssh_run() {
    local node="$1"
    shift
    ssh -n "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}

OUT_DIR="${OUT_DIR:-$SWEEP_TMP_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-preflight}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$SWEEP_TMP_DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] ||
    { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

docker_base=(
    docker run --rm --network host --ipc host
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render --cap-add IPC_LOCK
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    --security-opt seccomp=unconfined
)
[[ -z "${HOST_RDMA_LIB:-}" ]] ||
    docker_base+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")

run_token="$(date -u +%Y%m%dT%H%M%S)-$$"
active_nodes=()
active_names=()
cleanup() {
    set +e
    for index in "${!active_nodes[@]}"; do
        ssh_run "${active_nodes[index]}" docker rm -f "${active_names[index]}" \
            >/dev/null 2>&1
    done
    set -e
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_rank() {
    local node="$1" rank="$2" world="$3" name="$4"
    ssh_run "$node" timeout "${PREFLIGHT_TIMEOUT:-1800}" \
        "${docker_base[@]}" --name "$name" \
        -e "SLURM_PROCID=$rank" -e "SLURM_NNODES=$world" \
        -e "SLURMD_NODENAME=$node" -e "PREFLIGHT_HOST=$node" \
        -e "PREFLIGHT_IMAGE=$IMAGE" \
        -e "INFERA_PREFLIGHT_RUN_ID=$run_token" \
        -e "INFERA_PREFLIGHT_MOONCAKE_OPCODE=${PREFLIGHT_MOONCAKE_OPCODE:-write}" \
        -v "$OUT_DIR:$OUT_DIR" \
        "$IMAGE" python3 -m infera.tools.preflight \
        --dump-path "$OUT_DIR" --network --mooncake
}

world="${#nodes[@]}"
pids=()
for rank in "${!nodes[@]}"; do
    node="${nodes[rank]}"
    name="infera-preflight-$run_token-rank$rank"
    active_nodes+=("$node")
    active_names+=("$name")
    echo "starting rank $rank/$world on $node"
    run_rank "$node" "$rank" "$world" "$name" \
        >"$OUT_DIR/rank-$rank-$node.log" 2>&1 &
    pids+=("$!")
done

status=0
for rank in "${!pids[@]}"; do
    if ! wait "${pids[rank]}"; then
        echo "rank $rank (${nodes[rank]}) failed; see $OUT_DIR/rank-$rank-${nodes[rank]}.log" >&2
        status=1
    fi
done
cleanup
active_nodes=()
active_names=()
(( status == 0 )) || exit 1

echo "preflight artifacts: $OUT_DIR"
