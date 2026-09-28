#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?set CONFIG to frozen case file}"
export CONFIG
set -a; source "$CONFIG"; set +a
state() { date -u --iso-8601=seconds | tr '\n' ' ' > "$RUN/STATUS"; echo "$*" >> "$RUN/STATUS"; }
state SAMPLING
python3 "$TRACE_RUNTIME/scripts/sample_engine_metrics.py" --endpoint "prefill=http://$PREFILL_IP:29001/metrics" --endpoint "decode=http://$DECODE_IP:29002/metrics" --endpoint "router=http://$PREFILL_IP:28000/metrics" --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler.log" 2>&1 &
engine_pid=$!
python3 "$TRACE_RUNTIME/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler.log" 2>&1 &
node_pid=$!
(while true; do bash "$TRACE_RUNTIME/scripts/capture_live.sh" || true; sleep 60; done) > "$RUN/logs/capture.log" 2>&1 &
capture_pid=$!
printf '%s %s %s\n' "$engine_pid" "$node_pid" "$capture_pid" > "$RUN/monitor-pids.txt"
stop_observers() { kill -TERM "$engine_pid" "$node_pid" "$capture_pid" 2>/dev/null || true; }
trap stop_observers EXIT
state BENCHMARK_C80
date -u --iso-8601=ns > "$RUN/c80-started.txt"
if bash "$BENCH_DIR/agentx_bench.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" CONC=80 DURATION=3600 "OUT_DIR=$RUN/c80" > "$RUN/logs/c80.log" 2>&1; then
 date -u --iso-8601=ns > "$RUN/c80-completed.txt"
 state MEASUREMENT_COMPLETE
else
 state NEEDS_REVIEW_SERVICES_PRESERVED
 exit 1
fi
stop_observers
bash "$TRACE_RUNTIME/scripts/capture_live.sh"
python3 "$TRACE_RUNTIME/scripts/capture_final.py"
state READY_FOR_ANALYSIS
