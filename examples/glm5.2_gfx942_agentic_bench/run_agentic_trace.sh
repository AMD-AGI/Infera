#!/usr/bin/env bash
# Replay the agentic trace through the router with SGLang's built-in benchmark,
# then rescore it. Run on the prefill node inside the container.
#
#   NUM_PROMPTS=60 CONC=4 bash run_agentic_trace.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

NUM_PROMPTS="${NUM_PROMPTS:-60}"   # conversations, not requests
CONC="${CONC:-4}"
TAG="${1:-c${CONC}_n${NUM_PROMPTS}}"
DETAILS="$RESULT_DIR/agentic_${TAG}.jsonl"

[[ -f "$TRACE" ]] || { echo "[agentic] missing trace: $TRACE (build it first, see README)" >&2; exit 1; }

# Blocks left by an earlier run inflate the hit rate. flush_cache is a no-op while
# requests are in flight, so the fleet must be idle here.
curl -sf -X POST "$PREFILL_URL/flush_cache" >/dev/null || true
curl -sf -X POST "$DECODE_URL/flush_cache" >/dev/null || true
sleep 5

# --output-file appends, so a stale file would break the single-JSON scorer input.
rm -f "$DETAILS"

# --warmup-requests defaults to 1, which in multi-turn mode replays a whole
# conversation and pre-warms the cache the run is trying to measure.
python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --host "$PREFILL_IP" --port "$ROUTER_PORT" \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name agentic-trace --dataset-path "$TRACE" \
  --sharegpt-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
  --warmup-requests "${WARMUP:-0}" \
  --cache-report --output-details --output-file "$DETAILS" \
  ${EXTRA_BENCH_ARGS:-} 2>&1 | tee "$LOG_DIR/agentic_${TAG}.log"

# The tool's own input/cache summary is wrong in multi-turn mode; this is the
# number to read.
echo
python3 "$HERE/score_agentic_trace.py" "$TRACE" "$DETAILS" "$NUM_PROMPTS" ${SCORE_ARGS:-} \
  | tee "$RESULT_DIR/agentic_${TAG}.score.txt"
