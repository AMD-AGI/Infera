#!/usr/bin/env bash
# Purpose: Start etcd, topology workers, and the router; leave the stack running.
# Usage: ./launch.sh [KEY=VALUE ...]
# Artifacts: health/discovery JSON, server-info snapshots, and persistent worker logs.
# Artifact paths: OUT_DIR, default results/<UTC>-launch.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
overrides=("$@")

for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] ||
        { echo "expected KEY=VALUE, got '$assignment'" >&2; exit 2; }
    export "$assignment"
done
CONFIG="${CONFIG:-$DIR/config.sh}"
[[ -r "$CONFIG" ]] || { echo "config is not readable: $CONFIG" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
set -a
source "$CONFIG"
set +a
TOPOLOGY="${TOPOLOGY:-$DIR/topology.tsv}"
[[ -r "$TOPOLOGY" ]] || { echo "topology is not readable: $TOPOLOGY" >&2; exit 1; }
TOPOLOGY="$(cd "$(dirname "$TOPOLOGY")" && pwd)/$(basename "$TOPOLOGY")"
REMOTE_BENCH_DIR="${REMOTE_BENCH_DIR:-$DIR}"
: "${IMAGE:?set IMAGE}"
: "${MODEL:?set MODEL}"
: "${CONTROL_NODE:?set CONTROL_NODE}"

python3 "$DIR/tools/topology.py" validate "$TOPOLOGY"

rows() {
    local index instance role node ip gpus
    while IFS=$'\t' read -r index instance role node ip; do
        if [[ "$role" == prefill ]]; then
            gpus="$PREFILL_GPU_DEVICES"
        else
            gpus="$DECODE_GPU_DEVICES"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n' \
            "$instance" "$role" "$node" "$ip" "$gpus" \
            "$((ENGINE_PORT_BASE + index))" "$((BOOTSTRAP_PORT_BASE + index))" \
            "$((KV_EVENT_PORT_BASE + index))" "$((SNAPSHOT_PORT_BASE + index))" \
            "$CONTAINER_PREFIX-$instance"
    done < <(python3 "$DIR/tools/topology.py" rows "$TOPOLOGY")
}
read -r -a ssh_args <<< "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
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
    ssh "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}

OUT_DIR="${OUT_DIR:-$DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-launch}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$DIR/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] ||
    { echo "output path already exists: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR/failures" "$OUT_DIR/server-info" "$OUT_DIR/server-logs"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

control_ip="$(python3 "$DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
etcd_endpoint="$control_ip:$ETCD_PORT"
router_url="http://$control_ip:$ROUTER_PORT"
etcd_container="$CONTAINER_PREFIX-etcd"
router_container="$CONTAINER_PREFIX-router"
echo "control=$CONTROL_NODE etcd=$etcd_endpoint router=$router_url"

ssh_run "$CONTROL_NODE" docker run -d --init \
    --name "$etcd_container" --network host \
    "${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}" etcd \
    --advertise-client-urls "http://$etcd_endpoint" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT"
for attempt in $(seq 1 60); do
    ssh_run "$CONTROL_NODE" docker exec "$etcd_container" \
        etcdctl --endpoints "http://127.0.0.1:$ETCD_PORT" endpoint health \
        >/dev/null 2>&1 && break
    [[ "$attempt" != 60 ]] || { echo "etcd health check failed" >&2; exit 1; }
    sleep 1
done

wait_args=(
    python3 "$DIR/tools/wait_healthy.py"
    --ssh-options "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
    --timeout "${READY_TIMEOUT:-3600}" --interval "${HEALTH_INTERVAL:-10}"
    --probe-timeout "${HEALTH_PROBE_TIMEOUT:-5}"
    --failure-dir "$OUT_DIR/failures"
    --summary "$OUT_DIR/worker-health.json"
)
while IFS=$'\t' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    echo "starting $instance on $node"
    ssh_run "$node" bash "$REMOTE_BENCH_DIR/engine.sh" \
        "$role" "$instance" "$ip" "$gpus" "$engine" "$bootstrap" \
        "$kv" "$snapshot" "$container" "$etcd_endpoint" \
        "${overrides[@]}" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
        "SERVER_LOG=$OUT_DIR/server-logs/$instance.log" </dev/null
    wait_args+=(--target "$instance" "$node" "$container" "http://$ip:$engine/health")
done < <(rows)
"${wait_args[@]}"

router_args=(
    python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT"
    --router-backend "${ROUTER_BACKEND:-rust}"
    --discovery-backend etcd --etcd-endpoint "$etcd_endpoint"
    --request-transport http --kv-event-transport zmq
    --router-tokenizer-path "$MODEL"
)
if [[ "$ENABLE_KV_AWARE" == 1 ]]; then
    router_args+=(
        --router-policy kv-aware
        --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT"
        --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT"
    )
else
    router_args+=(--router-policy round-robin)
fi
echo "starting router on $CONTROL_NODE"
ssh_run "$CONTROL_NODE" docker run -d --init \
    --name "$router_container" --network host \
    -v "$MODEL:$MODEL:ro" "$IMAGE" "${router_args[@]}"

python3 "$DIR/tools/wait_healthy.py" \
    --target router "$CONTROL_NODE" "$router_container" "$router_url/health" \
    --ssh-options "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}" \
    --timeout "${ROUTER_READY_TIMEOUT:-300}" \
    --interval "${ROUTER_HEALTH_INTERVAL:-5}" \
    --probe-timeout "${HEALTH_PROBE_TIMEOUT:-5}" \
    --failure-dir "$OUT_DIR/failures" --summary "$OUT_DIR/router-health.json"

curl -fsS --max-time 30 "$router_url/v1/workers" |
    python3 -m json.tool > "$OUT_DIR/workers.json"
expected="$(( $(python3 "$DIR/tools/topology.py" count "$TOPOLOGY" prefill) + $(python3 "$DIR/tools/topology.py" count "$TOPOLOGY" decode) ))"
python3 - "$OUT_DIR/workers.json" "$expected" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
workers = payload
if isinstance(payload, dict):
    workers = payload.get("workers") or payload.get("data") or payload.get("instances") or []
if not isinstance(workers, list) or len(workers) != int(sys.argv[2]):
    raise SystemExit(f"router returned {len(workers) if isinstance(workers, list) else 'invalid'} workers; expected {sys.argv[2]}")
PY

while IFS=$'\t' read -r instance role node ip gpus engine rest; do
    curl -fsS --max-time 30 "http://$ip:$engine/get_server_info" \
        > "$OUT_DIR/server-info/$instance.json" ||
        curl -fsS --max-time 30 "http://$ip:$engine/v1/server_info" \
            > "$OUT_DIR/server-info/$instance.json"
done < <(rows)
echo "router ready: $router_url"
echo "inspection files: $OUT_DIR"
echo "use ./stop.sh when finished"
