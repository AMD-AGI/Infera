#!/usr/bin/env bash
# Start one Prefill or Decode worker. launch.sh invokes this over SSH.
set -euo pipefail
COMPONENT=engine
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/lib/common.sh"

(( $# >= 10 )) || die \
    "usage: engine.sh ROLE INSTANCE DATA_IP GPUS ENGINE_PORT BOOTSTRAP_PORT KV_EVENT_PORT SNAPSHOT_PORT CONTAINER ETCD_ENDPOINT [VAR=value ...]"
role="$1"
instance="$2"
node_ip="$3"
gpus="$4"
engine_port="$5"
bootstrap_port="$6"
kv_event_port="$7"
snapshot_port="$8"
container="$9"
etcd_endpoint="${10}"
shift 10
[[ "$role" == prefill || "$role" == decode ]] || die "invalid role: $role"
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe container name"

# launch.sh forwards its config path and command-line overrides.
load_config "$@"
prefix="${role^^}"
role_value() {
    local name="${prefix}_$1"
    printf '%s\n' "${!name}"
}

tp="$(role_value TP)"
ep="$(role_value EP)"
dp="$(role_value DP)"
dpa="$(bool01 "$(role_value DPA)")"
chunk="$(role_value CHUNK_SIZE)"
max_running="$(role_value MAX_RUNNING)"
graph_bs="$(role_value GRAPH_MAX_BS)"
mem_fraction="$(role_value MEM_FRACTION)"
hicache="$(bool01 "$(role_value HICACHE)")"
mtp=0
[[ "$role" == decode ]] && mtp="$(bool01 "$DECODE_MTP")"

nic="${DATA_NIC:-}"
if [[ -z "$nic" ]]; then
    nic="$(ip -o -4 addr show | awk -v ip="$node_ip" '$4 ~ ("^" ip "/") {print $2; exit}')"
fi
[[ -n "$nic" ]] || die "cannot resolve the NIC for $node_ip"

docker_args=(
    docker run -d --init --name "$container"
    --network host --ipc host --shm-size "${SHM_SIZE:-32g}"
    --device /dev/kfd --device /dev/dri --device /dev/infiniband
    --group-add video --group-add render
    --cap-add SYS_PTRACE --cap-add IPC_LOCK
    --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536
    -v "$MODEL:$MODEL:ro"
)
if [[ -n "${HOST_RDMA_LIB:-}" ]]; then
    [[ -r "$HOST_RDMA_LIB" ]] || die "RDMA provider is not readable: $HOST_RDMA_LIB"
    docker_args+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}:ro")
fi
docker_args+=(
    -e "HIP_VISIBLE_DEVICES=$gpus"
    -e "SGLANG_HOST_IP=$node_ip" -e "HOST_IP=$node_ip"
    -e "SGLANG_LOCAL_IP_NIC=$nic" -e "GLOO_SOCKET_IFNAME=$nic"
    -e "MC_GID_INDEX=$MC_GID_INDEX"
    -e "MC_TE_FILTERS=$MC_TE_FILTERS"
    -e MC_DISABLE_HIP_TRANSPORT=1
    -e "MC_ENABLE_DEST_DEVICE_AFFINITY=$MC_ENABLE_DEST_DEVICE_AFFINITY"
    -e "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    -e "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE"
    -e NCCL_IB_DISABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1
    -e HSA_NO_SCRATCH_RECLAIM=1 -e SGLANG_USE_AITER=1
    -e SGLANG_OPT_USE_TOPK_V2=false -e PYTHONNOUSERSITE=1
    -e "SGLANG_TIMEOUT_KEEP_ALIVE=${SGLANG_TIMEOUT_KEEP_ALIVE:-900}"
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SAFETENSORS_FAST_GPU=1
    -e HIP_FORCE_DEV_KERNARG=1 -e PYTHONHASHSEED=0
    -e "SGLANG_DP_USE_GATHERV=$dpa"
    -e "INFERA_ENGINE_READY_TIMEOUT=$READY_TIMEOUT"
    -e "INFERA_NODEPORT_RANGE=$INFERA_NODEPORT_RANGE"
    -e "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
    -e "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${BOOTSTRAP_TIMEOUT:-1800}"
)
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
    --cuda-graph-max-bs "$graph_bs"
    --max-running-requests "$max_running"
    --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}"
    --reasoning-parser "${REASONING_PARSER:-glm45}"
    --tool-call-parser "${TOOL_CALL_PARSER:-glm47}"
    --dsa-topk-backend "${DSA_TOPK_BACKEND:-aiter}"
    --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope
    --enable-cache-report --enable-metrics
    --disaggregation-mode "$role"
    --disaggregation-transfer-backend "$KV_P2P_TRANSFER"
)
[[ "$dp" -gt 1 ]] && engine_args+=(--dp-size "$dp")
[[ "$dpa" == 1 ]] && engine_args+=(--enable-dp-attention)
[[ -n "${RDMA_DEVICE:-}" ]] && engine_args+=(--disaggregation-ib-device "$RDMA_DEVICE")
if [[ "$role" == prefill ]]; then
    engine_args+=(--disaggregation-bootstrap-port "$bootstrap_port")
    if [[ "$(bool01 "$ENABLE_KV_AWARE")" == 1 ]]; then
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
if [[ "$mtp" == 1 ]]; then
    engine_args+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "$DECODE_SPEC_STEPS"
        --speculative-eagle-topk "$DECODE_SPEC_TOPK"
        --speculative-num-draft-tokens "$DECODE_SPEC_DRAFT_TOKENS"
    )
fi
if [[ "$hicache" == 1 ]]; then
    engine_args+=(
        --enable-hierarchical-cache
        --hicache-ratio "$(role_value HICACHE_RATIO)"
        --hicache-write-policy "$(role_value HICACHE_WRITE_POLICY)"
        --hicache-io-backend "$(role_value HICACHE_IO_BACKEND)"
        --hicache-mem-layout "$(role_value HICACHE_MEM_LAYOUT)"
    )
fi
[[ -n "${CONTEXT_LENGTH:-}" ]] && engine_args+=(--context-length "$CONTEXT_LENGTH")

log "$instance role=$role ip=$node_ip port=$engine_port GPUs=$gpus TP=$tp EP=$ep DP=$dp DPA=$dpa HiCache=$hicache MTP=$mtp"
print_command "${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
"${docker_args[@]}" "$IMAGE" "${engine_args[@]}"
