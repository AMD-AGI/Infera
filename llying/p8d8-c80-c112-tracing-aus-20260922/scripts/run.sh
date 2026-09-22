#!/usr/bin/env bash
# Run on the Prefill control node. State and logs survive client disconnection.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/config/config.sh"; set +a
[[ ! -e "$RUN/STATUS" ]] || { echo "run exists: $RUN"; exit 1; }
mkdir -p "$RUN/logs" "$RUN/sampling" "$RUN/traces" "$RUN/snapshot"
state() { printf '%s %s\n' "$(date -u --iso-8601=seconds)" "$*" | tee "$RUN/STATUS"; }
failure() { code=$?; state "FAILED rc=$code line=$1; preserve service and artifacts"; bash "$ROOT/scripts/capture_live.sh" || true; exit "$code"; }
trap 'failure $LINENO' ERR
cp -a "$ROOT/config" "$ROOT/docker" "$ROOT/scripts" "$RUN/snapshot/"
read -r -a opts <<< "$SSH_OPTS"
state WAITING_GPU_RELEASE
python3 "$ROOT/scripts/wait_nodes_idle.py" "$PREFILL_NODE" "$DECODE_NODE" --timeout 5400 --interval 30 > "$RUN/logs/wait-idle.log"
expected=sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    actual=$(ssh "${opts[@]}" "$node" docker image inspect "$IMAGE" --format '{{.Id}}')
    [[ "$actual" == "$expected" ]]
done
printf '%s\n' "$expected" > "$RUN/snapshot/image-id.txt"
state STARTING_COLLECTOR
docker run -d --init --name "$CONTAINER_PREFIX-collector" --network host \
    -v "$ROOT:$ROOT" "$IMAGE" python3 "$ROOT/scripts/otlp_jsonl_collector.py" \
    --output "$RUN/traces/spans.jsonl" --ready-file "$RUN/traces/ready.json" > "$RUN/logs/collector-id.txt"
for i in $(seq 1 60); do [[ ! -s "$RUN/traces/ready.json" ]] || break; sleep 1; done
test -s "$RUN/traces/ready.json"
state LAUNCHING
bash "$BENCH_DIR/launch.sh" "CONFIG=$ROOT/config/config.sh" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$RUN/launch" > "$RUN/logs/launch.log" 2>&1
state SMOKE
python3 "$ROOT/scripts/smoke.py" --output "$RUN/smoke.json"
bash "$ROOT/scripts/capture_live.sh"
python3 "$ROOT/scripts/validate_smoke.py" "$RUN"
python3 "$ROOT/scripts/sample_engine_metrics.py" \
    --endpoint "prefill=http://$PREFILL_IP:29001/metrics" \
    --endpoint "decode=http://$DECODE_IP:29002/metrics" \
    --endpoint "router=http://$PREFILL_IP:28000/metrics" \
    --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler.log" 2>&1 &
engine_pid=$!
python3 "$ROOT/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" \
    --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler.log" 2>&1 &
node_pid=$!
(while true; do bash "$ROOT/scripts/capture_live.sh" || true; sleep 60; done) > "$RUN/logs/capture.log" 2>&1 &
capture_pid=$!
printf '%s\n' "$engine_pid $node_pid $capture_pid" > "$RUN/monitor-pids.txt"
sleep 8
python3 "$ROOT/scripts/validate_live_samples.py" --engine "$RUN/sampling/engine.jsonl" --nodes "$RUN/sampling/nodes.jsonl" > "$RUN/sampling/preflight.json"
for c in 80 112; do
    state "BENCHMARK_C$c"
    date -u --iso-8601=ns > "$RUN/c$c-started.txt"
    bash "$BENCH_DIR/agentx_bench.sh" "CONFIG=$ROOT/config/config.sh" "TOPOLOGY=$TOPOLOGY" \
        "CONC=$c" "DURATION=3600" "OUT_DIR=$RUN/c$c" > "$RUN/logs/c$c.log" 2>&1
    date -u --iso-8601=ns > "$RUN/c$c-completed.txt"
    bash "$ROOT/scripts/capture_live.sh"
done
state BENCHMARKS_COMPLETED_ANALYSIS_PENDING
kill -TERM "$engine_pid" "$node_pid" "$capture_pid" || true
wait "$engine_pid" "$node_pid" "$capture_pid" || true
sleep 10
bash "$ROOT/scripts/capture_live.sh"
# Preserve services until diagnostics have been audited. Collector stays live so
# late spans and in-flight/cancelled requests remain observable.
