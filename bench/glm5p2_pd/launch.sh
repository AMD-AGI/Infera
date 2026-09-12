#!/usr/bin/env bash
# Start etcd, every P/D worker, and the router. Leave the stack running.
set -euo pipefail
COMPONENT=launch
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
mkdir -p "$OUT_DIR/failures" "$OUT_DIR/server-info"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

print_topology
check_nodes_idle
mapfile -t nodes < <(topology_nodes)
record_image_ids "${nodes[@]}"

control_ip="$(node_ip "$CONTROL_NODE")"
etcd_endpoint="$control_ip:$ETCD_PORT"
router_base_url="http://$control_ip:$ROUTER_PORT"
etcd_container="$(service_container etcd)"
router_container="$(service_container router)"
log "control=$CONTROL_NODE etcd=$etcd_endpoint router=$router_base_url"

ssh_exec "$CONTROL_NODE" docker run -d --init \
    --name "$etcd_container" --network host \
    "${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}" etcd \
    --advertise-client-urls "http://$etcd_endpoint" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT"

for ((attempt = 1; attempt <= 60; attempt++)); do
    if ssh_exec "$CONTROL_NODE" docker exec "$etcd_container" \
        etcdctl --endpoints "http://127.0.0.1:$ETCD_PORT" endpoint health \
        >/dev/null 2>&1; then
        break
    fi
    (( attempt < 60 )) || die "etcd failed its health check"
    sleep 1
done

declare -a wait_args=(
    python3 "$DIR/tools/wait_healthy.py"
    --ssh-options "$SSH_OPTS"
    --timeout "$READY_TIMEOUT"
    --interval "$HEALTH_INTERVAL"
    --probe-timeout "${HEALTH_PROBE_TIMEOUT:-5}"
    --failure-dir "$OUT_DIR/failures"
    --summary "$OUT_DIR/worker-health.json"
)
remote_overrides=(
    "${CLI_ASSIGNMENTS[@]}"
    "CONFIG=$CONFIG_FILE"
    "TOPOLOGY=$TOPOLOGY_FILE"
)
while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    log "starting $instance on $node"
    ssh_exec "$node" bash "$REMOTE_BENCH_DIR/engine.sh" \
        "$role" "$instance" "$ip" "$gpus" "$engine" "$bootstrap" \
        "$kv" "$snapshot" "$container" "$etcd_endpoint" \
        "${remote_overrides[@]}" </dev/null
    wait_args+=(
        --target "$instance" "$node" "$container" "http://$ip:$engine/health"
    )
done < <(topology_rows)

"${wait_args[@]}"

router_args=(
    python3 -m infera.server
    --host 0.0.0.0 --port "$ROUTER_PORT"
    --router-backend "${ROUTER_BACKEND:-rust}"
    --discovery-backend etcd --etcd-endpoint "$etcd_endpoint"
    --request-transport http --kv-event-transport zmq
    --router-tokenizer-path "$MODEL"
)
if [[ "$(bool01 "$ENABLE_KV_AWARE")" == 1 ]]; then
    router_args+=(
        --router-policy kv-aware
        --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT"
        --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT"
    )
else
    router_args+=(--router-policy round-robin)
fi
log "starting router on $CONTROL_NODE"
ssh_exec "$CONTROL_NODE" docker run -d --init \
    --name "$router_container" --network host \
    -v "$MODEL:$MODEL:ro" "$IMAGE" "${router_args[@]}"

python3 "$DIR/tools/wait_healthy.py" \
    --target router "$CONTROL_NODE" "$router_container" "$router_base_url/health" \
    --ssh-options "$SSH_OPTS" \
    --timeout "$ROUTER_READY_TIMEOUT" \
    --interval "${ROUTER_HEALTH_INTERVAL:-5}" \
    --probe-timeout "${HEALTH_PROBE_TIMEOUT:-5}" \
    --failure-dir "$OUT_DIR/failures" \
    --summary "$OUT_DIR/router-health.json"

curl -fsS --max-time 30 "$router_base_url/v1/workers" \
    | python3 -m json.tool >"$OUT_DIR/workers.json"

while IFS=$'\x1f' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    base="http://$ip:$engine"
    if ! curl -fsS --max-time 30 "$base/get_server_info" \
        >"$OUT_DIR/server-info/$instance.json"; then
        curl -fsS --max-time 30 "$base/v1/server_info" \
            >"$OUT_DIR/server-info/$instance.json"
    fi
    python3 -m json.tool "$OUT_DIR/server-info/$instance.json" \
        >"$OUT_DIR/server-info/$instance.json.tmp"
    mv "$OUT_DIR/server-info/$instance.json.tmp" "$OUT_DIR/server-info/$instance.json"
done < <(topology_rows)

python3 - "$OUT_DIR/workers.json" "$(( $(topology_count prefill) + $(topology_count decode) ))" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
workers = payload
if isinstance(payload, dict):
    workers = payload.get("workers") or payload.get("data") or payload.get("instances") or []
if not isinstance(workers, list) or len(workers) != int(sys.argv[2]):
    raise SystemExit(
        f"router discovery returned {len(workers) if isinstance(workers, list) else 'invalid'} "
        f"workers, expected {sys.argv[2]}"
    )
PY

log "PASS: router=$router_base_url"
log "inspection files=$OUT_DIR"
log "the stack is running; use ./stop.sh when finished"
