#!/usr/bin/env bash
set -a
source /perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924/config/run.sh
set +a
python3 "$TRACE_RUNTIME/scripts/sample_engine_metrics.py" --endpoint "prefill=http://$PREFILL_IP:29001/metrics" --endpoint "decode=http://$DECODE_IP:29002/metrics" --endpoint "router=http://$PREFILL_IP:28000/metrics" --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler-resumed.log" 2>&1 &
e=$!
python3 "$TRACE_RUNTIME/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler-resumed.log" 2>&1 &
n=$!
trap 'kill -TERM "$e" "$n" 2>/dev/null || true' EXIT
while [[ ! -e "$RUN/c80-completed.txt" && ! -e "$TRACE_RUNTIME/events/cleanup-complete" ]]; do sleep 5; done
