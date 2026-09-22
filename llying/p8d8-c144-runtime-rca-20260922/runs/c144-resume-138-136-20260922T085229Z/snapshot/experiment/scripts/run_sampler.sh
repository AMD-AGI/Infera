#!/usr/bin/env bash
# Run synchronized engine/rank and node/rail sampling for one experiment run.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.c144.p8d8.sh"
source "$CONFIG" || exit 1

RUN="${1:?usage: $0 RUN_DIR}"
DURATION="${DURATION:-0}"
ENGINE_INTERVAL="${ENGINE_INTERVAL:-2}"
NODE_INTERVAL="${NODE_INTERVAL:-5}"
DATA="$RUN/sampling"
LOGS="$RUN/logs"

if [[ -e "$DATA" ]]; then
    echo "sampling output already exists: $DATA" >&2
    exit 1
fi
mkdir -p "$DATA/raw" "$DATA/live" "$LOGS"

if [[ ! -d "$RUN/snapshot" ]]; then
    "$ROOT/scripts/snapshot_experiment.sh" "$RUN" \
        >"$LOGS/snapshot.log" 2>&1 || exit 1
fi

declare -A URLS=(
    [router]="http://$PREFILL_IP:28000"
    [prefill]="http://$PREFILL_IP:29001"
    [decode]="http://$DECODE_IP:29002"
)

for name in router prefill decode; do
    curl -fsS --max-time 30 "${URLS[$name]}/metrics" \
        >"$DATA/raw/$name-metrics-before.prom" || {
            echo "$name metrics endpoint is unavailable" >&2
            exit 1
        }
done
for name in prefill decode; do
    curl -fsS --max-time 30 "${URLS[$name]}/get_server_info" \
        >"$DATA/raw/$name-server-info.json" ||
        curl -fsS --max-time 30 "${URLS[$name]}/v1/server_info" \
            >"$DATA/raw/$name-server-info.json" || {
                echo "$name server_info endpoint is unavailable" >&2
                exit 1
            }
done
curl -fsS --max-time 30 "${URLS[router]}/v1/workers" \
    >"$DATA/raw/router-workers.json" || {
        echo "router worker inventory is unavailable" >&2
        exit 1
    }

ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker inspect "${CONTAINER_PREFIX}-prefill-0" \
    >"$DATA/raw/prefill-container.json" 2>&1 || exit 1
ssh -o BatchMode=yes "$DECODE_NODE" \
    docker inspect "${CONTAINER_PREFIX}-decode-0" \
    >"$DATA/raw/decode-container.json" 2>&1 || exit 1
ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker inspect "${CONTAINER_PREFIX}-router" \
    >"$DATA/raw/router-container.json" 2>&1 || exit 1

python3 "$ROOT/scripts/sample_engine_metrics.py" \
    --endpoint "router=${URLS[router]}" \
    --endpoint "prefill=${URLS[prefill]}" \
    --endpoint "decode=${URLS[decode]}" \
    --output "$DATA/live/engine-metrics.jsonl" \
    --interval "$ENGINE_INTERVAL" --duration "$DURATION" \
    >"$LOGS/engine-sampler.log" 2>&1 &
engine_pid=$!

python3 "$ROOT/scripts/sample_node_runtime.py" \
    --node "prefill=$PREFILL_NODE" \
    --node "decode=$DECODE_NODE" \
    --output "$DATA/live/node-runtime.jsonl" \
    --interval "$NODE_INTERVAL" --duration "$DURATION" \
    >"$LOGS/node-sampler.log" 2>&1 &
node_pid=$!

printf '%s\n' "$engine_pid" >"$DATA/engine-sampler.pid"
printf '%s\n' "$node_pid" >"$DATA/node-sampler.pid"
date -u --iso-8601=ns >"$DATA/sampling-started-at.txt"

signal_rc=0
stop_children() {
    kill "$engine_pid" "$node_pid" >/dev/null 2>&1 || true
    wait "$engine_pid" >/dev/null 2>&1 || true
    wait "$node_pid" >/dev/null 2>&1 || true
}
handle_int() {
    signal_rc=130
    stop_children
}
handle_term() {
    signal_rc=143
    stop_children
}
trap handle_int INT
trap handle_term TERM
trap stop_children EXIT

# Do not start a multi-hour benchmark with a silently empty or aggregate-only
# sampler. Two engine intervals and one node interval are enough for a smoke
# gate while the service is idle.
sleep 6
if ! kill -0 "$engine_pid" "$node_pid" >/dev/null 2>&1; then
    echo "a sampler exited during the smoke window" >&2
    exit 1
fi
python3 "$ROOT/scripts/validate_live_samples.py" \
    --engine "$DATA/live/engine-metrics.jsonl" \
    --nodes "$DATA/live/node-runtime.jsonl" \
    >"$DATA/live/preflight.json" || {
        cat "$DATA/live/preflight.json" >&2
        exit 1
    }

wait "$engine_pid"
engine_rc=$?
wait "$node_pid"
node_rc=$?
trap - EXIT INT TERM

for name in router prefill decode; do
    curl -fsS --max-time 30 "${URLS[$name]}/metrics" \
        >"$DATA/raw/$name-metrics-after.prom" || true
done
date -u --iso-8601=ns >"$DATA/sampling-completed-at.txt"
printf 'engine_sampler=%s\nnode_sampler=%s\n' "$engine_rc" "$node_rc" \
    >"$DATA/sampler-exit-codes.txt"

if (( signal_rc != 0 )); then
    exit "$signal_rc"
fi
if (( engine_rc != 0 || node_rc != 0 )); then
    echo "sampler failed: engine=$engine_rc node=$node_rc" >&2
    exit 1
fi
echo "sampling captured at $DATA"
