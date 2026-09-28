#!/usr/bin/env bash
set -euo pipefail
set -a; source /perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/config/performance.sh; set +a
python3 "$TRACE_RUNTIME/scripts/sample_engine_metrics.py" --endpoint "prefill=http://$PREFILL_IP:29001/metrics" --endpoint "decode=http://$DECODE_IP:29002/metrics" --endpoint "router=http://$PREFILL_IP:28000/metrics" --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler-recovered.log" 2>&1 &
a=$!
python3 "$TRACE_RUNTIME/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler-recovered.log" 2>&1 &
b=$!
printf '%s %s\n' "$a" "$b" > "$RUN/recovered-monitor-pids.txt"
wait "$a" "$b"
