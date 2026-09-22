#!/usr/bin/env bash
# Purpose: Start one role-specific SGLang Prefill or Decode worker container.
# Usage: ./engine.sh ROLE INSTANCE IP GPUS ENGINE_PORT BOOTSTRAP_PORT \
#   KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT [KEY=VALUE ...]
# Artifacts: detached worker container, persistent AITER JIT cache, optional log.
# Artifact paths: AITER cache defaults to /tmp/aiter-jit-<uid>/<image-id>;
#   SERVER_LOG selects the persistent log path.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

(( $# >= 10 )) || {
    echo "usage: $0 ROLE INSTANCE IP GPUS ENGINE_PORT BOOTSTRAP_PORT KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT [KEY=VALUE]" >&2
    exit 2
}
role="$1"; instance="$2"; node_ip="$3"; gpus="$4"
engine_port="$5"; bootstrap_port="$6"; kv_event_port="$7"; snapshot_port="$8"
container="$9"; etcd_endpoint="${10}"
shift 10
[[ "$role" == prefill || "$role" == decode ]] ||
    { echo "invalid role: $role" >&2; exit 2; }
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    { echo "unsafe container name: $container" >&2; exit 2; }

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
set -a
source "$CONFIG"
set +a
: "${IMAGE:?set IMAGE}"
: "${MODEL:?set MODEL}"
: "${SERVED_MODEL:?set SERVED_MODEL}"
: "${RDMA_DEVICE:?set RDMA_DEVICE}"

case "$role" in
    prefill)
        tp="$PREFILL_TP"; ep="$PREFILL_EP"; dp="$PREFILL_DP"; dpa="$PREFILL_DPA"
        chunk="$PREFILL_CHUNK_SIZE"; max_running="$PREFILL_MAX_RUNNING"
        graph_bs="$PREFILL_GRAPH_MAX_BS"; mem_fraction="$PREFILL_MEM_FRACTION"
        graph_bs_option="--cuda-graph-max-bs-prefill"
        hicache="$PREFILL_HICACHE"; hicache_ratio="$PREFILL_HICACHE_RATIO"
        hicache_policy="$PREFILL_HICACHE_WRITE_POLICY"
        hicache_backend="$PREFILL_HICACHE_IO_BACKEND"
        hicache_layout="$PREFILL_HICACHE_MEM_LAYOUT"
        scratch_reclaim="$PREFILL_HSA_NO_SCRATCH_RECLAIM"
        extra_args="${PREFILL_EXTRA_ARGS:-}"
        extra_env="${PREFILL_EXTRA_ENV:-}"
        ;;
    decode)
        tp="$DECODE_TP"; ep="$DECODE_EP"; dp="$DECODE_DP"; dpa="$DECODE_DPA"
        chunk="$DECODE_CHUNK_SIZE"; max_running="$DECODE_MAX_RUNNING"
        graph_bs="$DECODE_GRAPH_MAX_BS"; mem_fraction="$DECODE_MEM_FRACTION"
        graph_bs_option="--cuda-graph-max-bs-decode"
        hicache="$DECODE_HICACHE"; hicache_ratio="$DECODE_HICACHE_RATIO"
        hicache_policy="$DECODE_HICACHE_WRITE_POLICY"
        hicache_backend="$DECODE_HICACHE_IO_BACKEND"
        hicache_layout="$DECODE_HICACHE_MEM_LAYOUT"
        scratch_reclaim="$DECODE_HSA_NO_SCRATCH_RECLAIM"
        extra_args="${DECODE_EXTRA_ARGS:-}"
        extra_env="${DECODE_EXTRA_ENV:-}"
        ;;
esac
for integer in tp ep dp chunk max_running graph_bs; do
    [[ "${!integer}" =~ ^[1-9][0-9]*$ ]] ||
        { echo "$integer must be a positive integer" >&2; exit 2; }
done
[[ "$dpa" == 0 || "$dpa" == 1 ]] || { echo "DPA must be 0 or 1" >&2; exit 2; }
[[ "$hicache" == 0 || "$hicache" == 1 ]] ||
    { echo "HiCache must be 0 or 1" >&2; exit 2; }
[[ "$scratch_reclaim" == 0 || "$scratch_reclaim" == 1 ]] ||
    { echo "HSA_NO_SCRATCH_RECLAIM must be 0 or 1" >&2; exit 2; }
if [[ "$role" == decode && "$hicache" == 1 && "$DECODE_MTP" == 1 ]]; then
    echo "Decode MTP and HiCache cannot both be enabled" >&2
    exit 2
fi

nic="${DATA_NIC:-}"
if [[ -z "$nic" ]]; then
    nic="$(ip -o -4 addr show | awk -v wanted="$node_ip" '$4 ~ ("^" wanted "/") {print $2; exit}')"
fi
[[ -n "$nic" ]] || { echo "cannot resolve NIC for $node_ip" >&2; exit 1; }

image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
image_key="${image_id#sha256:}"
[[ "$image_key" =~ ^[0-9a-f]{64}$ ]] ||
    { echo "cannot derive image ID for AITER cache: $image_id" >&2; exit 1; }
aiter_jit_cache="${AITER_JIT_CACHE_ROOT:-/tmp/aiter-jit-$(id -u)}/$image_key"
mkdir -p "$aiter_jit_cache"

docker_args=(
    docker run -d --init --name "$container"
    --network host --ipc host --shm-size "${SHM_SIZE:-32g}"
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render
    --cap-add SYS_PTRACE --cap-add IPC_LOCK
    --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    -v "$MODEL:$MODEL:ro"
    -v "$aiter_jit_cache:/aiter-jit"
    -e AITER_JIT_DIR=/aiter-jit
)
if [[ -n "${HOST_RDMA_LIB:-}" ]]; then
    [[ -r "$HOST_RDMA_LIB" ]] ||
        { echo "RDMA provider is not readable: $HOST_RDMA_LIB" >&2; exit 1; }
    docker_args+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")
fi
docker_args+=(
    -e "HIP_VISIBLE_DEVICES=$gpus"
    -e "SGLANG_HOST_IP=$node_ip" -e "HOST_IP=$node_ip"
    -e "SGLANG_LOCAL_IP_NIC=$nic" -e "GLOO_SOCKET_IFNAME=$nic"
    -e "MC_GID_INDEX=$MC_GID_INDEX" -e "MC_TE_FILTERS=$MC_TE_FILTERS"
    -e "MC_DISABLE_HIP_TRANSPORT=$MC_DISABLE_HIP_TRANSPORT"
    -e "MC_ENABLE_DEST_DEVICE_AFFINITY=$MC_ENABLE_DEST_DEVICE_AFFINITY"
    -e "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    -e "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" -e NCCL_IGNORE_CPU_AFFINITY=1
    -e "HSA_NO_SCRATCH_RECLAIM=$scratch_reclaim"
    -e SGLANG_USE_AITER=1
    -e "SGLANG_OPT_USE_TOPK_V2=$SGLANG_OPT_USE_TOPK_V2"
    -e "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=$SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"
    -e PYTHONNOUSERSITE=1
    -e "SGLANG_TIMEOUT_KEEP_ALIVE=$SGLANG_TIMEOUT_KEEP_ALIVE"
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SAFETENSORS_FAST_GPU=1
    -e HIP_FORCE_DEV_KERNARG=1 -e PYTHONHASHSEED=0
    -e "SGLANG_DP_USE_GATHERV=$dpa"
    -e "INFERA_ENGINE_READY_TIMEOUT=$READY_TIMEOUT"
    -e "INFERA_NODEPORT_RANGE=$INFERA_NODEPORT_RANGE"
    -e "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
    -e "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
    -e "SGLANG_ENABLE_FAILED_SESSION_PROBE=$SGLANG_ENABLE_FAILED_SESSION_PROBE"
    -e "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=$SGLANG_FAILED_SESSION_PROBE_INTERVAL_S"
)
# Role-scoped escape hatch for one-off container env during bring-up and
# debugging, so a config that differs from the default recipe does not need this
# script edited. Space-separated KEY=VALUE items; empty by default, so unused it
# is a no-op.
if [[ -n "${extra_env:-}" ]]; then
    read -r -a extra_env_items <<< "$extra_env"
    for item in "${extra_env_items[@]}"; do
        [[ "$item" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
            { echo "invalid EXTRA_ENV item: $item" >&2; exit 2; }
        docker_args+=(-e "$item")
    done
fi
if [[ "$role" == decode && -n "${DECODE_SIMULATE_ACC_LEN:-}" ]]; then
    docker_args+=(
        -e "SGLANG_SIMULATE_ACC_LEN=$DECODE_SIMULATE_ACC_LEN"
        -e SGLANG_SIMULATE_ACC_METHOD=match-expected
        -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    )
fi

engine_args=(
    python3 -m infera.engine.sglang
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL"
    --host "$node_ip" --port "$engine_port" --advertise-host "$node_ip"
    --etcd-endpoint "$etcd_endpoint" --discovery-backend etcd
    --request-transport http --kv-event-transport zmq
    --tp-size "$tp" --ep-size "$ep" --trust-remote-code
    --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
    --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
    --mem-fraction-static "$mem_fraction"
    --chunked-prefill-size "$chunk"
    "$graph_bs_option" "$graph_bs"
    --max-running-requests "$max_running"
    --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}"
    --reasoning-parser "${REASONING_PARSER:-glm45}"
    --tool-call-parser "${TOOL_CALL_PARSER:-glm47}"
    --enable-fused-qk-norm-rope
    --enable-cache-report --enable-metrics
    --disaggregation-mode "$role"
    --disaggregation-transfer-backend "$KV_P2P_TRANSFER"
)
[[ "$dp" -gt 1 ]] && engine_args+=(--dp-size "$dp")
[[ "$dpa" == 1 ]] && engine_args+=(--enable-dp-attention)
[[ -n "${DSA_TOPK_BACKEND:-}" ]] &&
    engine_args+=(--dsa-topk-backend "$DSA_TOPK_BACKEND")
[[ -n "${JSON_MODEL_OVERRIDE_ARGS:-}" ]] &&
    engine_args+=(--json-model-override-args "$JSON_MODEL_OVERRIDE_ARGS")
[[ -n "$RDMA_DEVICE" ]] && engine_args+=(--disaggregation-ib-device "$RDMA_DEVICE")
if [[ "$role" == prefill ]]; then
    engine_args+=(--disaggregation-bootstrap-port "$bootstrap_port")
    if [[ "$ENABLE_KV_AWARE" == 1 ]]; then
        engine_args+=(
            --enable-kv-events --kv-events on
            --kv-events-bind "tcp://0.0.0.0:$kv_event_port"
            --kv-snapshot-port "$snapshot_port"
        )
    else
        engine_args+=(--no-enable-kv-events --kv-events off)
    fi
else
    engine_args+=(--no-enable-kv-events --kv-events off)
fi
if [[ "$role" == decode && "$DECODE_MTP" == 1 ]]; then
    engine_args+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$DECODE_SPEC_STEPS"
        --speculative-eagle-topk "$DECODE_SPEC_TOPK"
        --speculative-num-draft-tokens "$DECODE_SPEC_DRAFT_TOKENS"
    )
fi
if [[ "$hicache" == 1 ]]; then
    engine_args+=(
        --enable-hierarchical-cache --hicache-ratio "$hicache_ratio"
        --hicache-write-policy "$hicache_policy"
        --hicache-io-backend "$hicache_backend"
        --hicache-mem-layout "$hicache_layout"
    )
fi
# AITER all-reduce fusion. Emitted by default, so this is a no-op unless a config
# sets AITER_ALLREDUCE_FUSION=0. Made conditional because sglang's CLI exposes
# only the positive `--enable-aiter-allreduce-fusion` (store_true, no `--no-`
# form), so there is otherwise no way to A/B it; the only code path that turns it
# off is --enable-deterministic-inference, which changes much more besides.
[[ "${AITER_ALLREDUCE_FUSION:-1}" == 1 ]] &&
    engine_args+=(--enable-aiter-allreduce-fusion)
[[ -n "${CONTEXT_LENGTH:-}" ]] && engine_args+=(--context-length "$CONTEXT_LENGTH")
# Role-scoped escape hatch for one-off engine flags during bring-up and
# debugging, so an experiment does not have to edit this script. Word-split on
# whitespace; quote nothing fancier than that. Empty by default.
if [[ -n "${extra_args:-}" ]]; then
    read -r -a extra_arg_items <<< "$extra_args"
    engine_args+=("${extra_arg_items[@]}")
fi

echo "$instance role=$role ip=$node_ip port=$engine_port GPUs=$gpus TP=$tp EP=$ep DP=$dp DPA=$dpa HiCache=$hicache"
echo "AITER JIT cache: $aiter_jit_cache"
printf ' + %q' "${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
echo
"${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
if [[ -n "${SERVER_LOG:-}" ]]; then
    mkdir -p "$(dirname "$SERVER_LOG")"
    nohup docker logs --timestamps -f "$container" \
        >"$SERVER_LOG" 2>&1 </dev/null &
    echo "server log: $SERVER_LOG"
fi
