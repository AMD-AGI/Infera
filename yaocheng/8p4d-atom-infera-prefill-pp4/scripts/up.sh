#!/usr/bin/env bash
# Start etcd, the prefill instances (TP1 x PP4 each) and the decode engine
# (TP4 + DCP4), then the router, for one CONC.
# Usage: scripts/up.sh [CONC=48] [MTP_AL=] [RUN_ID=...] [KEY=VALUE ...]
source "$(dirname "$0")/common.sh" "$@"

RUN_DIR="$TMP_DIR/runs/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-c$CONC}"
mkdir -p "$RUN_DIR/logs"
echo "$RUN_DIR" > "$TMP_DIR/current_run"

# Store live logs on the control node, including when shared NFS is read-only.
follow() {
    nohup bash -c '
        common="$1" node="$2" name="$3"
        set --
        source "$common"
        on "$node" docker logs -f --timestamps "$name"
    ' _ "$KIT_DIR/scripts/common.sh" "$1" "$PREFIX-$2" \
        > "$RUN_DIR/logs/$2.live.log" 2>&1 < /dev/null &
}
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    echo "$node amdgpu $(on "$node" cat /sys/module/amdgpu/version)"
    wait_vram_free "$node"
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

spec=(--method mtp --num-speculative-tokens "$MTP_K")
[[ -z "$MTP_AL" ]] || spec+=(--spec-decode-acceptance-length "$MTP_AL")

# engine NAME NODE IP GPUS ARGS...: one infera.engine.atom container; NAME is
# the role (prefill or decode), optionally followed by an instance number.
engine() {
    local name="$1" node="$2" ip="$3" gpus="$4" role="${1%%[0-9]*}" image_id env_args=()
    local role_env="${role^^}_ENV" extra_env="${role^^}_EXTRA_ENV" util="${role^^}_MEM_UTIL"
    shift 4
    image_id="$(on "$node" docker image inspect -f '{{.Id}}' "$IMAGE")"
    for kv in "${ENGINE_ENV[@]}" ${!role_env} ${!extra_env}; do env_args+=(-e "$kv"); done
    on "$node" docker run -d --name "$PREFIX-$name" --network host --ipc host --shm-size 32g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
        -v "$MODEL:$MODEL:ro" -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro \
        -v "$PREFIX-cache-$name-${image_id:7:12}:/root/.cache" \
        -e "HIP_VISIBLE_DEVICES=$gpus" -e "ATOM_HOST_IP=$ip" "${env_args[@]}" "$IMAGE" \
        python3 -m infera.engine.atom --etcd-endpoint "$etcd" --advertise-host "$ip" \
        --model "$MODEL" --host 0.0.0.0 --trust-remote-code --kv_cache_dtype fp8 \
        --block-size "$BLOCK_SIZE" --enable_prefix_caching --online_quant_config "$QUANT_CONFIG" \
        "${spec[@]}" --timeout-keep-alive 900 --gpu-memory-utilization "${!util}" "$@"
    follow "$node" "$name"
}

for i in "${!PREFILL_GROUPS[@]}"; do
    port=$((PREFILL_PORT + 2 * i))
    kv="$(kv_config kv_producer $((PREFILL_HANDSHAKE_PORT + 20 * i)) "$port" "$PREFILL_IP")"
    if (( PREFILL_OFFLOAD_GB > 0 )); then
        kv="{\"kv_connector\":\"multi\",\"connectors\":[$kv,{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}]}"
    fi
    engine "prefill$i" "$PREFILL_NODE" "$PREFILL_IP" "${PREFILL_GROUPS[i]}" \
        -tp 1 -pp 4 --enforce-eager --max-num-seqs 512 \
        --max-num-batched-tokens "$PREFILL_MAX_TOKENS" --server-port "$port" \
        --kv-transfer-config "$kv" $PREFILL_EXTRA_ARGS
done

cudagraph="[1,2,4,8"
for ((size = 12; size <= 2 * CONC; size += 4)); do cudagraph+=",$size"; done
engine decode "$DECODE_NODE" "$DECODE_IP" "$DECODE_GPUS" \
    -tp 4 --decode-context-parallel-size 4 \
    --cudagraph-capture-sizes "$cudagraph]" --max-num-seqs $((2 * CONC)) \
    --max-num-batched-tokens "$DECODE_MAX_TOKENS" --server-port "$DECODE_PORT" \
    --kv-transfer-config "$(kv_config kv_consumer "$DECODE_HANDSHAKE_PORT" "$DECODE_PORT" "$DECODE_IP")" \
    $DECODE_EXTRA_ARGS

for i in "${!PREFILL_GROUPS[@]}"; do
    wait_ready "$PREFILL_NODE" "$PREFIX-prefill$i" "http://$PREFILL_IP:$((PREFILL_PORT + 2 * i))/health"
done
wait_ready "$DECODE_NODE" "$PREFIX-decode" "http://$DECODE_IP:$DECODE_PORT/health"

on "$CONTROL_NODE" docker run -d --name "$PREFIX-router" --network host \
    -v "$MODEL:$MODEL:ro" "$IMAGE" python3 -m infera.server \
    --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend python \
    --discovery-backend etcd --etcd-endpoint "$etcd" --request-transport http \
    --kv-event-transport zmq --router-policy round-robin --router-tokenizer-path "$MODEL"
follow "$CONTROL_NODE" router
wait_ready "$CONTROL_NODE" "$PREFIX-router" "http://$CONTROL_IP:$ROUTER_PORT/health"
curl -s "http://$CONTROL_IP:$ROUTER_PORT/v1/workers" | tee "$RUN_DIR/workers.json"
echo
echo "up: CONC=$CONC MTP_K=$MTP_K MTP_AL=${MTP_AL:-off} prefill=${#PREFILL_GROUPS[@]}xPP4 run=$RUN_DIR"
