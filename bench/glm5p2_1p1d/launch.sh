#!/usr/bin/env bash
# Bring up etcd, one prefill leg, one decode leg, then the Infera router.
# Usage: bash launch.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

ETCD_ENDPOINT="$PREFILL_IP:$ETCD_PORT"

on() {
    local node="$1"
    shift
    ssh $SSH_OPTS "$node" "$@"
}

remove_containers() {
    local node="$1" reason="$2" container state
    shift 2
    for container in "$@"; do
        if state="$(on "$node" docker inspect \
            -f 'status={{.State.Status}},exit_code={{.State.ExitCode}},oom_killed={{.State.OOMKilled}}' \
            "$container" 2>/dev/null)"; then
            echo "[launch] removing $container on $node ($reason; $state)"
            if [[ "$state" == status=running,* ]]; then
                on "$node" docker stop --time "$CONTAINER_STOP_TIMEOUT" \
                    "$container" >/dev/null || true
            fi
            on "$node" docker rm "$container" >/dev/null ||
                on "$node" docker rm -f "$container" >/dev/null
        fi
    done
}

wait_for_idle_gpus() {
    local node="$1" vram="" deadline=$((SECONDS + GPU_IDLE_TIMEOUT))
    while true; do
        vram="$(
            on "$node" rocm-smi --showmemuse 2>/dev/null \
                | awk -F': ' '/VRAM%/ {n++; if ($NF + 0 > max) max=$NF + 0} END {if (n) print max + 0}'
        )"
        [[ -n "$vram" ]] || {
            echo "[launch] cannot read all-card VRAM usage on $node" >&2
            return 1
        }
        (( vram <= 10 )) && return 0
        (( SECONDS < deadline )) || {
            echo "[launch] $node still busy after ${GPU_IDLE_TIMEOUT}s (max VRAM=${vram}%)" >&2
            return 1
        }
        echo "[launch] waiting for $node HBM release (max VRAM=${vram}%)"
        sleep 30
    done
}

declare -A image_ids=()
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    image_ids["$node"]="$(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" || {
        echo "[launch] image not found on $node: $IMAGE" >&2
        exit 1
    }
done
# A shared tag pointing at different builds silently invalidates the curve.
[[ "${image_ids[$PREFILL_NODE]}" == "${image_ids[$DECODE_NODE]}" ]] || {
    echo "[launch] image id mismatch for $IMAGE" >&2
    echo "[launch]   $PREFILL_NODE: ${image_ids[$PREFILL_NODE]}" >&2
    echo "[launch]   $DECODE_NODE: ${image_ids[$DECODE_NODE]}" >&2
    exit 1
}
echo "[launch] image $IMAGE -> ${image_ids[$PREFILL_NODE]} on both nodes"

echo "[launch] 0/4 stopping the previous experiment deployment"
remove_containers "$PREFILL_NODE" "replace before RUN_ID=$RUN_ID" \
    glm52-pd-router glm52-pd-prefill glm52-pd-etcd
remove_containers "$DECODE_NODE" "replace before RUN_ID=$RUN_ID" \
    glm52-pd-decode
sleep "${RESTART_SETTLE_SECONDS:-5}"
wait_for_idle_gpus "$PREFILL_NODE"
wait_for_idle_gpus "$DECODE_NODE"

echo "[launch] 1/4 etcd on $PREFILL_NODE ($ETCD_ENDPOINT)"
on "$PREFILL_NODE" docker run -d --init --name glm52-pd-etcd --network host \
    quay.io/coreos/etcd:v3.5.14 \
    etcd --advertise-client-urls "http://$ETCD_ENDPOINT" \
         --listen-client-urls "http://0.0.0.0:$ETCD_PORT" >/dev/null

for attempt in $(seq 1 30); do
    if on "$PREFILL_NODE" docker exec glm52-pd-etcd \
        etcdctl --endpoints="http://127.0.0.1:$ETCD_PORT" endpoint health \
        >/dev/null 2>&1; then
        break
    fi
    [[ "$attempt" -lt 30 ]] || { echo "[launch] etcd did not become healthy" >&2; exit 1; }
    sleep 1
done

common_env=(
    "ETCD_ENDPOINT=$ETCD_ENDPOINT"
    "IMAGE=$IMAGE"
    "MODEL=$MODEL"
    "SERVED_MODEL=$SERVED_MODEL"
    "PREFILL_PORT=$PREFILL_PORT"
    "DECODE_PORT=$DECODE_PORT"
    "BOOTSTRAP_PORT=$BOOTSTRAP_PORT"
    "INFERA_NODEPORT_RANGE=$INFERA_NODEPORT_RANGE"
    "PREFILL_TP_SIZE=$PREFILL_TP_SIZE"
    "DECODE_TP_SIZE=$DECODE_TP_SIZE"
    "PREFILL_EP_SIZE=$PREFILL_EP_SIZE"
    "DECODE_EP_SIZE=$DECODE_EP_SIZE"
    "PREFILL_DP_SIZE=$PREFILL_DP_SIZE"
    "DECODE_DP_SIZE=$DECODE_DP_SIZE"
    "CONTEXT_LENGTH=$CONTEXT_LENGTH"
    "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
    "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
    "CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS"
    "PREFILL_MEM_FRACTION=$PREFILL_MEM_FRACTION"
    "DECODE_MEM_FRACTION=$DECODE_MEM_FRACTION"
    "PREFILL_DPA=$PREFILL_DPA"
    "DECODE_DPA=$DECODE_DPA"
    "ENABLE_MTP=$ENABLE_MTP"
    "SPEC_STEPS=$SPEC_STEPS"
    "SPEC_DRAFT_TOKENS=$SPEC_DRAFT_TOKENS"
    "SPEC_TOPK=$SPEC_TOPK"
    "SIMULATE_ACC_LEN=$SIMULATE_ACC_LEN"
    "ENABLE_KV_AWARE=$ENABLE_KV_AWARE"
    "KV_PUB_PORT=$KV_PUB_PORT"
    "KV_SNAPSHOT_PORT=$KV_SNAPSHOT_PORT"
    "ENABLE_HICACHE=$ENABLE_HICACHE"
    "PREFILL_ENABLE_HICACHE=$PREFILL_ENABLE_HICACHE"
    "DECODE_ENABLE_HICACHE=$DECODE_ENABLE_HICACHE"
    "HICACHE_RATIO=$HICACHE_RATIO"
    "HICACHE_WRITE_POLICY=$HICACHE_WRITE_POLICY"
    "HICACHE_IO_BACKEND=$HICACHE_IO_BACKEND"
    "HICACHE_MEM_LAYOUT=$HICACHE_MEM_LAYOUT"
    "ENABLE_KVD=$ENABLE_KVD"
    "KV_P2P_TRANSFER=$KV_P2P_TRANSFER"
    "RDMA_DEVICE=$RDMA_DEVICE"
    "MC_GID_INDEX=$MC_GID_INDEX"
    "MOONCAKE_DISABLE_HIP_DMABUF=$MOONCAKE_DISABLE_HIP_DMABUF"
    "MC_ENABLE_DEST_DEVICE_AFFINITY=$MC_ENABLE_DEST_DEVICE_AFFINITY"
    "MC_TE_FILTERS=${MC_TE_FILTERS:-$RDMA_DEVICE}"
    "RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE"
    "HOST_RDMA_LIB=$HOST_RDMA_LIB"
    "HOST_RDMA_MOUNT=$HOST_RDMA_MOUNT"
    "NOFILE_ULIMIT=$NOFILE_ULIMIT"
    "READY_TIMEOUT=$READY_TIMEOUT"
)

echo "[launch] 2/4 prefill on $PREFILL_NODE and decode on $DECODE_NODE"
on "$PREFILL_NODE" env "${common_env[@]}" \
    ROLE=prefill NODE_IP="$PREFILL_IP" bash "$DIR/engine.sh"
on "$DECODE_NODE" env "${common_env[@]}" \
    ROLE=decode NODE_IP="$DECODE_IP" bash "$DIR/engine.sh"

wait_leg() {
    local node="$1" container="$2" ip="$3" port="$4" state
    for attempt in $(seq 1 "$LEG_HEALTH_TRIES"); do
        if on "$node" docker exec "$container" \
            curl -sf -m 5 "http://$ip:$port/health" >/dev/null 2>&1; then
            echo "[launch] $container healthy after $(((attempt - 1) * 15))s"
            return 0
        fi
        if ! state="$(on "$node" docker inspect \
            -f 'running={{.State.Running}},status={{.State.Status}},exit_code={{.State.ExitCode}},oom_killed={{.State.OOMKilled}},error={{.State.Error}},finished_at={{.State.FinishedAt}}' \
            "$container" 2>/dev/null)"; then
            echo "[launch] $container missing or inspect failed on $node" >&2
            return 1
        fi
        if [[ "$state" != running=true,* ]]; then
            echo "[launch] $container not running on $node: $state" >&2
            if [[ "$state" == *"exit_code=137"* && "$state" == *"oom_killed=false"* ]]; then
                echo "[launch] exit 137 without OOM is consistent with external SIGKILL; caller unknown" >&2
            fi
            on "$node" docker logs "$container" || true
            return 1
        fi
        sleep 15
    done
    echo "[launch] timeout waiting for $container" >&2
    on "$node" docker logs "$container" || true
    return 1
}

echo "[launch] 3/4 waiting for both legs; first load can take 10-20 minutes"
wait_leg "$PREFILL_NODE" glm52-pd-prefill "$PREFILL_IP" "$PREFILL_PORT"
wait_leg "$DECODE_NODE" glm52-pd-decode "$DECODE_IP" "$DECODE_PORT"

echo "[launch] 4/4 router on $PREFILL_NODE"
router_args=(
    python3 -m infera.server
    --host 0.0.0.0 --port "$ROUTER_PORT"
    --router-backend rust
    --discovery-backend etcd --etcd-endpoint "$ETCD_ENDPOINT"
    --request-transport http --kv-event-transport zmq
    --router-tokenizer-path "$MODEL"
)
if [[ "$ENABLE_KV_AWARE" == "1" ]]; then
    router_args+=(
        --router-policy kv-aware
        --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT"
        --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT"
    )
else
    router_args+=(--router-policy round-robin)
fi

on "$PREFILL_NODE" docker run -d --init --name glm52-pd-router --network host \
    -v "$MODEL:$MODEL:ro" "$IMAGE" "${router_args[@]}" >/dev/null

for attempt in $(seq 1 60); do
    if curl -sf -m 5 "http://$PREFILL_IP:$ROUTER_PORT/health" >/dev/null 2>&1; then
        echo "[launch] ready: http://$PREFILL_IP:$ROUTER_PORT"
        curl -sS "http://$PREFILL_IP:$ROUTER_PORT/v1/workers"
        echo
        exit 0
    fi
    [[ "$attempt" -lt 60 ]] || {
        echo "[launch] router did not become healthy" >&2
        on "$PREFILL_NODE" docker logs glm52-pd-router || true
        exit 1
    }
    sleep 5
done
