#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/config/config.sh"; set +a
archive="$RUN/recovery/stale-sampler-preflight"
mkdir -p "$archive"
for file in engine.jsonl nodes.jsonl preflight.json; do
    [[ ! -f "$RUN/sampling/$file" ]] || mv "$RUN/sampling/$file" "$archive/$file"
done
python3 "$ROOT/scripts/sample_engine_metrics.py" \
    --endpoint prefill=http://10.235.192.136:29001/metrics \
    --endpoint decode=http://10.235.192.128:29002/metrics \
    --endpoint router=http://10.235.192.136:28000/metrics \
    --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler.log" 2>&1 &
engine_pid=$!
python3 "$ROOT/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" \
    --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler.log" 2>&1 &
node_pid=$!
trap 'kill -TERM "$engine_pid" "$node_pid" 2>/dev/null || true' EXIT
echo "$engine_pid $node_pid" > "$RUN/recovered-sampler-pids.txt"
sleep 10
python3 "$ROOT/scripts/validate_live_samples.py" --engine "$RUN/sampling/engine.jsonl" \
    --nodes "$RUN/sampling/nodes.jsonl" > "$RUN/sampling/preflight.json"
while ! grep -qE 'FAILED|BENCHMARKS_COMPLETED' "$RUN/STATUS"; do
    kill -0 "$engine_pid" "$node_pid"
    sleep 20
done
