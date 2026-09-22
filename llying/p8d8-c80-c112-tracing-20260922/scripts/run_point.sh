#!/usr/bin/env bash
# Run one traced AgentX point against an already-live traced service.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.trace.p8d8.sh"
TOPOLOGY="$ROOT/config/topology.trace.tsv"
TOOLS="/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c144-runtime-rca-20260922/scripts"
source "$CONFIG" || exit 1

POINT="${1:?usage: $0 CONC OUTPUT_DIR}"
OUT="${2:?usage: $0 CONC OUTPUT_DIR}"
[[ "$POINT" =~ ^(80|112)$ ]] || { echo "point must be 80 or 112" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "output exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs" "$OUT/sampling/live" "$OUT/sampling/raw" "$OUT/server-logs"
OUT="$(cd "$OUT" && pwd)"
date -u --iso-8601=ns >"$OUT/started-at.txt"

declare -A URLS=(
    [router]="http://$PREFILL_IP:$ROUTER_PORT"
    [prefill]="http://$PREFILL_IP:$ENGINE_PORT_BASE"
    [decode]="http://$DECODE_IP:$((ENGINE_PORT_BASE + 1))"
)
for name in router prefill decode; do
    curl -fsS --max-time 30 "${URLS[$name]}/metrics" \
        >"$OUT/sampling/raw/$name-metrics-before.prom" || exit 1
done

bash "$ROOT/scripts/collector_ctl.sh" start "$OUT/traces" \
    >"$OUT/logs/collector-start.log" 2>&1 || exit 1

python3 "$TOOLS/sample_engine_metrics.py" \
    --endpoint "router=${URLS[router]}" \
    --endpoint "prefill=${URLS[prefill]}" \
    --endpoint "decode=${URLS[decode]}" \
    --output "$OUT/sampling/live/engine-metrics.jsonl" \
    --interval 2 --duration 0 >"$OUT/logs/engine-sampler.log" 2>&1 &
engine_pid=$!
python3 "$TOOLS/sample_node_runtime.py" \
    --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" \
    --output "$OUT/sampling/live/node-runtime.jsonl" \
    --interval 5 --duration 0 >"$OUT/logs/node-sampler.log" 2>&1 &
node_pid=$!

stop_monitors() {
    kill -TERM "$engine_pid" "$node_pid" >/dev/null 2>&1 || true
    wait "$engine_pid" >/dev/null 2>&1 || true
    wait "$node_pid" >/dev/null 2>&1 || true
}
cleanup() {
    stop_monitors
    bash "$ROOT/scripts/collector_ctl.sh" stop >/dev/null 2>&1 || true
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

sleep 7
kill -0 "$engine_pid" "$node_pid" >/dev/null 2>&1 || {
    echo "sampler failed during startup" >&2
    cleanup
    exit 1
}
python3 "$TOOLS/validate_live_samples.py" \
    --engine "$OUT/sampling/live/engine-metrics.jsonl" \
    --nodes "$OUT/sampling/live/node-runtime.jsonl" \
    >"$OUT/sampling/live/preflight.json" || {
        cleanup
        exit 1
    }

bash "$BENCH_DIR/agentx_bench.sh" \
    "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    "CONC=$POINT" "DURATION=3600" \
    "AGENTX_WARMUP_REQUESTS_PER_LANE=10" \
    "OUT_DIR=$OUT/bench" "AGENTX_CACHE_DIR=$ROOT/cache/agentx" \
    >"$OUT/logs/bench-console.log" 2>&1
bench_rc=$?
echo "$bench_rc" >"$OUT/bench-exit-code.txt"

stop_monitors
bash "$ROOT/scripts/collector_ctl.sh" stop \
    >"$OUT/logs/collector-stop.log" 2>&1 || true
trap - INT TERM
date -u --iso-8601=ns >"$OUT/completed-at.txt"

ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker logs --timestamps "${CONTAINER_PREFIX}-prefill-0" \
    >"$OUT/server-logs/prefill.log" 2>&1 || true
ssh -o BatchMode=yes "$DECODE_NODE" \
    docker logs --timestamps "${CONTAINER_PREFIX}-decode-0" \
    >"$OUT/server-logs/decode.log" 2>&1 || true

start="$(<"$OUT/started-at.txt")"
end="$(<"$OUT/completed-at.txt")"
python3 "$ROOT/scripts/parse_req_time_stats.py" \
    --log "prefill=$OUT/server-logs/prefill.log" \
    --log "decode=$OUT/server-logs/decode.log" \
    --start "$start" --end "$end" \
    --output "$OUT/request-time-stats.jsonl" \
    >"$OUT/logs/parse-time-stats.log" 2>&1 || true
python3 "$ROOT/scripts/analyze_otlp_traces.py" \
    --input "$OUT/traces/spans.jsonl" \
    --output-dir "$OUT/trace-analysis" --point "c$POINT" \
    >"$OUT/logs/analyze-traces.log" 2>&1 || true

if (( bench_rc == 0 )); then
    echo "COMPLETED" >"$OUT/STATUS"
else
    echo "FAILED: preserve service and inspect" >"$OUT/STATUS"
fi
exit "$bench_rc"
