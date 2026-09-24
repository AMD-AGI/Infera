#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
ROOT="$TRACE_RUNTIME"
state() { printf '%s %s\n' "$(date -u --iso-8601=seconds)" "$*" | tee "$RUN/STATUS"; }
failure() { code=$?; state "FAILED rc=$code line=$1"; bash "$ROOT/scripts/capture_live.sh" || true; exit "$code"; }
trap 'failure $LINENO' ERR
if [[ "${1:-}" != --after-smoke ]]; then
state VALIDATING_CASE
python3 "$ROOT/scripts/validate_allocation.py"
python3 "$ROOT/scripts/validate_case.py"
state SMOKE
python3 "$ROOT/scripts/smoke.py" --output "$RUN/smoke.json"
python3 "$ROOT/scripts/stream_smoke.py"
python3 "$ROOT/scripts/verify_guard_mode.py"
python3 "$ROOT/scripts/verify_experiment.py"
bash "$ROOT/scripts/capture_live.sh"
date +%s > "$RUN/snapshot/flush-start-epoch.txt"
python3 "$ROOT/scripts/validate_smoke.py" "$RUN"
python3 "$ROOT/scripts/check_empty.py"
else
    [[ -s "$RUN/smoke-validation.json" && -s "$RUN/snapshot/router-mode-validation.json" && ! -e "$RUN/c80-started.txt" ]]
    state RECOVERING_AFTER_HOST_GAUGE_AUDIT
    python3 "$ROOT/scripts/validate_allocation.py"
    date +%s > "$RUN/snapshot/flush-start-epoch.txt"
    python3 "$ROOT/scripts/validate_smoke.py" "$RUN"
    python3 "$ROOT/scripts/check_empty.py"
fi
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then
    state SMOKE_COMPLETE
    exit 0
fi
python3 "$ROOT/scripts/sample_engine_metrics.py" --endpoint "prefill=http://$PREFILL_IP:29001/metrics" --endpoint "decode=http://$DECODE_IP:29002/metrics" --endpoint "router=http://$PREFILL_IP:28000/metrics" --output "$RUN/sampling/engine.jsonl" --interval 2 --duration 0 > "$RUN/logs/engine-sampler.log" 2>&1 &
engine_pid=$!
python3 "$ROOT/scripts/sample_node_runtime.py" --node "prefill=$PREFILL_NODE" --node "decode=$DECODE_NODE" --output "$RUN/sampling/nodes.jsonl" --interval 5 --duration 0 > "$RUN/logs/node-sampler.log" 2>&1 &
node_pid=$!
(while true; do bash "$ROOT/scripts/capture_live.sh"; sleep 60; done) > "$RUN/logs/capture.log" 2>&1 &
capture_pid=$!
python3 "$ROOT/scripts/gpu_ownership_watch.py" > "$RUN/logs/gpu-ownership.log" 2>&1 &
owner_pid=$!
printf '%s\n' "$engine_pid $node_pid $capture_pid $owner_pid" > "$RUN/monitor-pids.txt"
sleep 8
kill -0 "$engine_pid" "$node_pid" "$capture_pid"
python3 "$ROOT/scripts/validate_live_samples.py" --engine "$RUN/sampling/engine.jsonl" --nodes "$RUN/sampling/nodes.jsonl" > "$RUN/sampling/preflight.json"
python3 "$ROOT/scripts/validate_allocation.py"
[[ ! -e "$RUN/INVALID" ]]
state BENCHMARK_C80
date -u --iso-8601=ns > "$RUN/c80-started.txt"
bash "$BENCH_DIR/agentx_bench.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" CONC=80 DURATION=3600 "OUT_DIR=$RUN/c80" > "$RUN/logs/c80.log" 2>&1
date -u --iso-8601=ns > "$RUN/c80-completed.txt"
kill -TERM "$engine_pid" "$node_pid" "$capture_pid" "$owner_pid" || true
wait "$engine_pid" "$node_pid" "$capture_pid" "$owner_pid" || true
sleep 10
bash "$ROOT/scripts/capture_live.sh"
state ANALYZING
python3 "$ROOT/scripts/analyze.py" "$RUN"
python3 "$ROOT/scripts/analyze_runtime.py" "$RUN"
python3 "$ROOT/scripts/analyze_spans.py" "$RUN"
python3 "$ROOT/scripts/analyze_chunk_evidence.py" "$RUN" "$RUN/analysis"
python3 "$ROOT/scripts/compare_c80_runs.py" "$BASELINE_RUN" "$RUN" --reference-label reference --candidate-label "$RUN_ID" --output-dir "$RUN/analysis/comparison"
state COMPLETE_REVIEW_PENDING
