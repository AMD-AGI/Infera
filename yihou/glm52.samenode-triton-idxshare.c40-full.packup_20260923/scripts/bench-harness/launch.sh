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
        # ENGINE_PORT_STRIDE: same-node only. SGLang derives an INTERNAL port
        # block from --port (server_args.py:836,851-861): a fixed
        # ZMQ_TCP_PORT_DELTA offset, then it reserves
        # port_base+0..NUM_DERIVED_PORTS-1, where NUM_DERIVED_PORTS is 6, or
        # 6+dp_size on the rust path. With two legs on ONE host and engine ports
        # one apart (29001/29002) those blocks OVERLAP, and the second leg dies:
        #   zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:29236')
        #   DetokenizerManager.init_ipc_channels -> sigquit -> exited with code -9
        # Observed first-hand, rounds/006-bringup-1. Stride the ENGINE port only
        # (the bootstrap/kv/snapshot bases are already distinct and feed nothing
        # derived). Defaults to 1, so every cross-node deployment is unchanged.
        printf '%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n' \
            "$instance" "$role" "$node" "$ip" "$gpus" \
            "$((ENGINE_PORT_BASE + index * ${ENGINE_PORT_STRIDE:-1}))" \
            "$((BOOTSTRAP_PORT_BASE + index))" \
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
# Same-node start gate. On a single-host 1P1D the two legs' RCCL communicators
# initialise concurrently and race; one dies with
#   rccl .../p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument
#   NCCL error: unhandled cuda error
# It is a RACE, not a config error: swapping the GPU sets between the legs makes
# the FAILING leg follow start ORDER, not the GPU set or the role (rounds 7/8 vs
# 9 vs 15 — same GPU assignment, different leg fails; see analysis note). Cross-
# node the two inits run on different machines and cannot race. Fix: before
# starting a row whose node ALREADY has a started leg, wait for that leg's /health
# (reusing wait_healthy.py) — the by-hand round-16 fix, made reproducible. This
# only fires when a node repeats in the topology, so every cross-node deployment
# is behaviourally unchanged. Knobs: SAME_NODE_START_GATE=health|off (default
# health); ENGINE_START_STAGGER_S (default 0) adds a fixed same-node sleep for
# operators who prefer a delay to a health poll.
declare -A started_on_node=()
gate_number=0
gate_same_node() {
    local node="$1"
    [[ -n "${started_on_node[$node]:-}" ]] || return 0   # no prior leg on this host
    local stagger="${ENGINE_START_STAGGER_S:-0}"
    if [[ "$stagger" != 0 ]]; then
        echo "same-node stagger: sleeping ${stagger}s before the next leg on $node"
        sleep "$stagger"
    fi
    [[ "${SAME_NODE_START_GATE:-health}" == health ]] || return 0
    echo "same-node gate: waiting for the already-started leg(s) on $node to answer /health"
    local gate_args=(
        python3 "$DIR/tools/wait_healthy.py"
        --ssh-options "${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
        --timeout "${READY_TIMEOUT:-3600}" --interval "${HEALTH_INTERVAL:-10}"
        --probe-timeout "${HEALTH_PROBE_TIMEOUT:-5}"
        --failure-dir "$OUT_DIR/failures"
        --summary "$OUT_DIR/gate-$gate_number-$node.json"
    )
    local record inst cont url
    while IFS= read -r record; do
        [[ -n "$record" ]] || continue
        IFS=$'\t' read -r inst cont url <<< "$record"
        gate_args+=(--target "$inst" "$node" "$cont" "$url")
    done <<< "${started_on_node[$node]}"
    gate_number=$((gate_number + 1))
    "${gate_args[@]}" ||
        { echo "same-node gate failed on $node; not starting the next leg" >&2; exit 1; }
}
while IFS=$'\t' read -r instance role node ip gpus engine bootstrap kv snapshot container; do
    gate_same_node "$node"
    echo "starting $instance on $node"
    ssh_run "$node" bash "$REMOTE_BENCH_DIR/engine.sh" \
        "$role" "$instance" "$ip" "$gpus" "$engine" "$bootstrap" \
        "$kv" "$snapshot" "$container" "$etcd_endpoint" \
        "${overrides[@]}" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
        "SERVER_LOG=$OUT_DIR/server-logs/$instance.log" </dev/null
    health_url="http://$ip:$engine/health"
    record="${instance}"$'\t'"${container}"$'\t'"${health_url}"
    if [[ -n "${started_on_node[$node]:-}" ]]; then
        started_on_node[$node]+=$'\n'"$record"
    else
        started_on_node[$node]="$record"
    fi
    wait_args+=(--target "$instance" "$node" "$container" "$health_url")
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
# The router's clap uses ArgAction::Set: INFERA_PD_DP_RANK_AFFINITY accepts only
# true/false, never 1/0, so translate the numeric knob here.
if [[ "$PD_DP_RANK_AFFINITY" == 1 ]]; then
    pd_dp_rank_affinity=true
else
    pd_dp_rank_affinity=false
fi
echo "starting router on $CONTROL_NODE"
ssh_run "$CONTROL_NODE" docker run -d --init \
    --name "$router_container" --network host \
    -e "INFERA_PD_DP_RANK_AFFINITY=$pd_dp_rank_affinity" \
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
