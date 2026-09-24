#!/usr/bin/env bash
# Start etcd, the prefill and decode engines, then the router, for one CONC.
# Usage: scripts/up.sh [CONC=16] [MTP_AL=] [RUN_ID=...] [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

RUN_DIR="$TMP_DIR/runs/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-c$CONC}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$TMP_DIR/current_run"
# amdgpu 6.19.x needs lower *_MEM_UTIL, see ../8p4d-atom-infera/.record/issues.md #10.
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    echo "$node amdgpu $(on "$node" cat /sys/module/amdgpu/version)"
done
etcd="$CONTROL_IP:$ETCD_PORT"

on "$CONTROL_NODE" docker run -d --name "$PREFIX-etcd" --network host "$ETCD_IMAGE" etcd \
    --advertise-client-urls "http://$etcd" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT"

# kv_config ROLE HANDSHAKE_PORT HTTP_PORT IP
kv_config() {
    echo "{\"kv_role\":\"$1\",\"kv_connector\":\"mooncake\",\"protocol\":\"rdma\",\"handshake_port\":$2,\"http_port\":$3,\"proxy_ip\":\"$4\"}"
}

# engine ROLE NODE IP GPUS ARGS...: one infera.engine.atom container.
engine() {
    local role="$1" node="$2" ip="$3" gpus="$4" image_id env_args=()
    local role_env="${role^^}_ENV" extra_env="${role^^}_EXTRA_ENV" util="${role^^}_MEM_UTIL"
    shift 4
    image_id="$(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE")"
    for kv in "${ENGINE_ENV[@]}" ${!role_env} ${!extra_env}; do env_args+=(-e "$kv"); done
    on "$node" docker run -d --name "$PREFIX-$role" --network host --ipc host --shm-size 32g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        -v "$MODEL:$MODEL:ro" -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro \
        -v "$PREFIX-cache-${image_id:7:12}:/root/.cache" \
        -e "HIP_VISIBLE_DEVICES=$gpus" -e "ATOM_HOST_IP=$ip" "${env_args[@]}" "$IMAGE" \
        python3 -m infera.engine.atom --etcd-endpoint "$etcd" --advertise-host "$ip" \
        --model "$MODEL" --host 0.0.0.0 --trust-remote-code --kv_cache_dtype fp8 \
        --block-size 16 --enable_prefix_caching --online_quant_config "$QUANT_CONFIG" \
        --method mtp --num-speculative-tokens "$MTP_K" --max-num-batched-tokens 16384 \
        --gpu-memory-utilization "${!util}" "$@"
}

prefill_kv="$(kv_config kv_producer "$PREFILL_HANDSHAKE_PORT" "$PREFILL_PORT" "$PREFILL_IP")"
if (( PREFILL_OFFLOAD_GB > 0 )); then
    prefill_kv="{\"kv_connector\":\"multi\",\"connectors\":[$prefill_kv,{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}"
fi
engine prefill "$PREFILL_NODE" "$PREFILL_IP" "$PREFILL_GPUS" \
    -tp 8 --enable-dp-attention $PREFILL_GRAPH_ARGS --max-num-seqs 512 --server-port "$PREFILL_PORT" \
    --kv-transfer-config "$prefill_kv" $PREFILL_EXTRA_ARGS

cudagraph="[1,2,4,8"
for ((size = 12; size <= 2 * CONC; size += 4)); do cudagraph+=",$size"; done
decode_args=(-tp 8 --decode-context-parallel-size 8
    --cudagraph-capture-sizes "$cudagraph]" --max-num-seqs $((2 * CONC))
    --server-port "$DECODE_PORT" --kv-transfer-config
    "$(kv_config kv_consumer "$DECODE_HANDSHAKE_PORT" "$DECODE_PORT" "$DECODE_IP")")
[[ -z "$MTP_AL" ]] || decode_args+=(--spec-decode-acceptance-length "$MTP_AL")
engine decode "$DECODE_NODE" "$DECODE_IP" "$DECODE_GPUS" "${decode_args[@]}" $DECODE_EXTRA_ARGS

wait_ready "$PREFILL_NODE" "$PREFIX-prefill" "http://$PREFILL_IP:$PREFILL_PORT/health"
wait_ready "$DECODE_NODE" "$PREFIX-decode" "http://$DECODE_IP:$DECODE_PORT/health"

on "$CONTROL_NODE" docker run -d --name "$PREFIX-router" --network host \
    -v "$MODEL:$MODEL:ro" "$IMAGE" python3 -m infera.server \
    --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend python \
    --discovery-backend etcd --etcd-endpoint "$etcd" --request-transport http \
    --kv-event-transport zmq --router-policy round-robin --router-tokenizer-path "$MODEL"
wait_ready "$CONTROL_NODE" "$PREFIX-router" "http://$CONTROL_IP:$ROUTER_PORT/health"
curl -s "http://$CONTROL_IP:$ROUTER_PORT/v1/workers" | tee "$RUN_DIR/workers.json"
echo
echo "up: CONC=$CONC MTP_K=$MTP_K MTP_AL=${MTP_AL:-off} run=$RUN_DIR"
