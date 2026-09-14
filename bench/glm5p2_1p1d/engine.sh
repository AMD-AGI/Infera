#!/usr/bin/env bash
# Launch one GLM-5.2 SGLang PD leg. Called remotely by launch.sh.
# Usage: ROLE=prefill|decode NODE_IP=<data-ip> bash engine.sh  # normally via launch.sh
set -euo pipefail

: "${ROLE:?set ROLE=prefill|decode}"
: "${NODE_IP:?set NODE_IP to this node data-plane IP}"
: "${ETCD_ENDPOINT:?set ETCD_ENDPOINT=host:port}"
: "${IMAGE:?set IMAGE}"
: "${MODEL:?set MODEL}"
: "${SERVED_MODEL:?set SERVED_MODEL}"
: "${RDMA_DEVICE:?set RDMA_DEVICE}"
MC_TE_FILTERS="${MC_TE_FILTERS:-$RDMA_DEVICE}"

case "$ROLE" in
    prefill)
        PORT="${PREFILL_PORT:?}"
        MEM_FRACTION="${PREFILL_MEM_FRACTION:?}"
        TP_SIZE="${PREFILL_TP_SIZE:?}"
        EP_SIZE="${PREFILL_EP_SIZE:?}"
        DP_SIZE="${PREFILL_DP_SIZE:?}"
        DPA="${PREFILL_DPA:?}"
        HICACHE_ENABLED="${PREFILL_ENABLE_HICACHE:?}"
        CONTAINER="glm52-pd-prefill"
        ;;
    decode)
        PORT="${DECODE_PORT:?}"
        MEM_FRACTION="${DECODE_MEM_FRACTION:?}"
        TP_SIZE="${DECODE_TP_SIZE:?}"
        EP_SIZE="${DECODE_EP_SIZE:?}"
        DP_SIZE="${DECODE_DP_SIZE:?}"
        DPA="${DECODE_DPA:?}"
        HICACHE_ENABLED="${DECODE_ENABLE_HICACHE:?}"
        CONTAINER="glm52-pd-decode"
        ;;
    *)
        echo "[engine] ROLE must be prefill or decode, got: $ROLE" >&2
        exit 64
        ;;
esac

if [[ "${ENABLE_KVD:?}" == "1" ]]; then
    echo "[engine] ENABLE_KVD=1 is reserved for the later HiCache A/B; baseline keeps it off" >&2
    exit 64
fi

NIC="${NIC:-$(ip -o -4 addr show | awk -v ip="$NODE_IP" '$4 ~ ("^" ip "/") {print $2; exit}')}"
[[ -n "$NIC" ]] || { echo "[engine] cannot find NIC for $NODE_IP" >&2; exit 1; }

ROLE_ARGS=(
    --disaggregation-mode "$ROLE"
    --disaggregation-transfer-backend "${KV_P2P_TRANSFER:?}"
    --disaggregation-ib-device "${RDMA_DEVICE:?}"
)
if [[ "$ROLE" == "prefill" ]]; then
    ROLE_ARGS+=(--disaggregation-bootstrap-port "${BOOTSTRAP_PORT:?}")
fi

PARALLEL_ARGS=(--ep-size "$EP_SIZE")
if (( DP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--dp-size "$DP_SIZE")
fi
if [[ "$DPA" == "1" ]]; then
    PARALLEL_ARGS+=(--enable-dp-attention)
fi

MTP_ARGS=()
if [[ "$ROLE" == "decode" && "${ENABLE_MTP:?}" == "1" ]]; then
    MTP_ARGS=(
        --speculative-algorithm EAGLE
        --speculative-num-steps "${SPEC_STEPS:?}"
        --speculative-eagle-topk "${SPEC_TOPK:?}"
        --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:?}"
    )
fi

HICACHE_ARGS=()
if [[ "$HICACHE_ENABLED" == "1" ]]; then
    HICACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "${HICACHE_RATIO:?}"
        --hicache-write-policy "${HICACHE_WRITE_POLICY:?}"
        --hicache-io-backend "${HICACHE_IO_BACKEND:?}"
        --hicache-mem-layout "${HICACHE_MEM_LAYOUT:?}"
    )
fi

SIMULATION_ENV=()
if [[ "$ROLE" == "decode" && -n "${SIMULATE_ACC_LEN:-}" ]]; then
    SIMULATION_ENV=(
        -e "SGLANG_SIMULATE_ACC_LEN=$SIMULATE_ACC_LEN"
        -e SGLANG_SIMULATE_ACC_METHOD=match-expected
        -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    )
fi

CONTEXT_ARGS=()
if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
    CONTEXT_ARGS=(--context-length "$CONTEXT_LENGTH")
fi

KV_EVENT_ARGS=(--no-enable-kv-events --kv-events off)
if [[ "${ENABLE_KV_AWARE:?}" == "1" && "$ROLE" == "prefill" ]]; then
    KV_EVENT_ARGS=(
        --enable-kv-events
        --kv-events on
        --kv-events-bind "tcp://0.0.0.0:${KV_PUB_PORT:-5557}"
        --kv-snapshot-port "${KV_SNAPSHOT_PORT:-8801}"
    )
fi

RDMA_MOUNT_ARGS=()
if [[ -n "${HOST_RDMA_LIB:-}" ]]; then
    [[ -r "$HOST_RDMA_LIB" ]] || {
        echo "[engine] host RDMA provider is not readable: $HOST_RDMA_LIB" >&2
        exit 1
    }
    RDMA_MOUNT_ARGS=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:?}:ro")
fi

echo "[engine] role=$ROLE node=$NODE_IP port=$PORT tp=$TP_SIZE ep=$EP_SIZE dp=$DP_SIZE dpa=$DPA mtp=${ENABLE_MTP:?}"
echo "[engine] image=$IMAGE model=$MODEL rdma=$RDMA_DEVICE gid=$MC_GID_INDEX nic=$NIC hicache=$HICACHE_ENABLED simulated_acc=${SIMULATE_ACC_LEN:-off}"
if [[ "${ENABLE_KV_AWARE:?}" == "1" && "$ROLE" == "prefill" ]]; then
    echo "[engine] kv-aware=on publisher=prefill events=tcp://0.0.0.0:${KV_PUB_PORT:-5557} snapshot_port=${KV_SNAPSHOT_PORT:-8801}"
elif [[ "${ENABLE_KV_AWARE:?}" == "1" ]]; then
    echo "[engine] kv-aware=on publisher=off-on-decode"
else
    echo "[engine] kv-aware=off"
fi

# Off when DPA=0. For supported TP=DP DP-attention layouts, this enables the
# all_gatherv + reduce_scatterv path for SUM_LEN prefill/extend collectives.
# Keep a tiny init as PID 1 so worker descendants are reaped during shutdown;
# without it, a dead engine can remain a zombie and pin all eight GPUs.
docker run -d --init --name "$CONTAINER" --network host --ipc host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    --ulimit nofile="${NOFILE_ULIMIT:?}" \
    "${RDMA_MOUNT_ARGS[@]}" \
    -v "$MODEL:$MODEL:ro" \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e SGLANG_HOST_IP="$NODE_IP" -e HOST_IP="$NODE_IP" \
    -e SGLANG_LOCAL_IP_NIC="$NIC" -e GLOO_SOCKET_IFNAME="$NIC" \
    -e MOONCAKE_DISABLE_HIP_DMABUF="${MOONCAKE_DISABLE_HIP_DMABUF:?}" \
    -e MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:?}" \
    -e MC_DISABLE_HIP_TRANSPORT=1 \
    -e MC_TE_FILTERS="$MC_TE_FILTERS" \
    -e MC_GID_INDEX="${MC_GID_INDEX:?}" \
    -e RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:?}" \
    -e NCCL_IB_DISABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_OPT_USE_TOPK_V2=false \
    -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e PYTHONNOUSERSITE=1 \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 \
    -e SAFETENSORS_FAST_GPU=1 -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONHASHSEED=0 \
    -e SGLANG_DP_USE_GATHERV="$DPA" \
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${BOOTSTRAP_TIMEOUT:-1800}" \
    -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${BOOTSTRAP_TIMEOUT:-1800}" \
    -e INFERA_ENGINE_READY_TIMEOUT="${READY_TIMEOUT:?}" \
    -e INFERA_NODEPORT_RANGE="${INFERA_NODEPORT_RANGE:?}" \
    "${SIMULATION_ENV[@]}" \
    "$IMAGE" \
    python3 -m infera.engine.sglang \
        --model-path "$MODEL" \
        --served-model-name "$SERVED_MODEL" \
        --host "$NODE_IP" --port "$PORT" --advertise-host "$NODE_IP" \
        --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
        --request-transport http --kv-event-transport zmq \
        --tp-size "$TP_SIZE" --trust-remote-code \
        --dsa-prefill-backend tilelang \
        --dsa-decode-backend tilelang \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static "$MEM_FRACTION" \
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}" \
        --reasoning-parser glm45 --tool-call-parser glm47 \
        --dsa-topk-backend aiter \
        --enable-aiter-allreduce-fusion \
        --enable-fused-qk-norm-rope \
        --enable-cache-report --enable-metrics \
        "${CONTEXT_ARGS[@]}" \
        "${KV_EVENT_ARGS[@]}" \
        "${PARALLEL_ARGS[@]}" \
        "${MTP_ARGS[@]}" \
        "${HICACHE_ARGS[@]}" \
        "${ROLE_ARGS[@]}"

echo "[engine] $CONTAINER started; inspect with: docker logs -f $CONTAINER"
