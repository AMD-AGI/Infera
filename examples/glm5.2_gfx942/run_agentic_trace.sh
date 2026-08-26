#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Replay the multi-turn agentic trace through a router and rescore the result.
#
# This is the workload the tuned recipe was chosen on. Unlike bench.sh's random
# dataset it has real prefix reuse, so it is the only one of the two that can say
# anything about the kv-aware router or the KV cache. See README §5.
#
# Runs wherever SGLang and the tokenizer are both available:
#
#   docker arm  inside the engine container on the prefill node
#                 NUM_PROMPTS=60 CONC=16 bash run_agentic_trace.sh
#
#   k8s arm     inside a client container, which bench_client.sh starts for you
#                 bash bench_client.sh k8s http://<router-pod-ip>:8000 \
#                      http://<prefill-node>:30001 http://<decode-node>:31501
#
# Args: [tag]. Everything else comes from env.sh or the environment:
#   ROUTER_URL   router to drive          (default: the docker arm's)
#   FLUSH_URLS   engine URLs to flush     (default: PREFILL_URL DECODE_URL)
#   TOKENIZER    local tokenizer path     (default: MODEL)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

NUM_PROMPTS="${NUM_PROMPTS:-60}"   # conversations, not requests
CONC="${CONC:-16}"
TAG="${1:-${TAG:-agentic}}"
NAME="${TAG}_c${CONC}_n${NUM_PROMPTS}"
DETAILS="$RESULT_DIR/${NAME}.jsonl"
TOKENIZER="${TOKENIZER:-$MODEL}"

read -r -a FLUSH_URL_ARR <<< "${FLUSH_URLS:-$PREFILL_URL $DECODE_URL}"

[[ -f "$TRACE" ]] || {
  echo "[agentic] missing trace: $TRACE" >&2
  echo "[agentic] build it first with weka_to_agentic_trace.py -- see README §5.2" >&2
  exit 1
}

curl -sf -m 10 "$ROUTER_URL/health" >/dev/null \
  || { echo "[agentic] router is not answering at $ROUTER_URL -- run verify.sh first" >&2; exit 1; }

# The name the router answers to is the engine's --model-path, which is a mount
# path and so differs between the arms (/models/GLM-5.2-FP8 under k8s). The
# tokenizer meanwhile has to be a path that exists HERE. The two cannot be assumed
# equal, and sending the wrong name is not a clean 404: the router replies
# `no active mixed worker for model="..."` and every request fails in
# milliseconds, which reads like a dead fleet rather than a name mismatch.
SERVED_MODEL="${SERVED_MODEL:-}"
if [[ -z "$SERVED_MODEL" ]]; then
  SERVED_MODEL="$(curl -sf -m 10 "$ROUTER_URL/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  [[ -n "$SERVED_MODEL" ]] \
    || { echo "[agentic] could not read the served model name from $ROUTER_URL/v1/models" >&2; exit 1; }
  echo "[agentic] served model name: $SERVED_MODEL"
fi

# The scorer's ideal is page-aligned, so a wrong page size silently moves the
# efficiency number rather than failing. Read it from the server instead of
# trusting a default -- SGLang forces 64 for GLM-5.2's DSA attention, but a
# deployment can still report something else.
PAGE_SIZE="${PAGE_SIZE:-}"
if [[ -z "$PAGE_SIZE" ]]; then
  PAGE_SIZE="$(curl -sf -m 10 "${FLUSH_URL_ARR[0]}/get_server_info" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("page_size",""))' 2>/dev/null || true)"
  [[ -n "$PAGE_SIZE" ]] \
    || { echo "[agentic] could not read page_size from ${FLUSH_URL_ARR[0]}; pass PAGE_SIZE=" >&2; exit 1; }
  echo "[agentic] page_size: $PAGE_SIZE"
fi

# Blocks left by an earlier run inflate the hit rate, and flush_cache is a no-op
# while requests are in flight -- it still returns success -- so the fleet has to
# be idle here. An efficiency ABOVE 100% is the signal that this did not take.
for u in "${FLUSH_URL_ARR[@]}"; do
  printf '[agentic] flush %s -> ' "$u"
  curl -sf -m 20 -X POST "$u/flush_cache" >/dev/null 2>&1 && echo ok || echo FAILED
done
sleep 5

# --output-file appends, so a stale file would break the single-JSON scorer input.
rm -f "$DETAILS"

echo "[agentic] $TAG router=$ROUTER_URL conc=$CONC convs=$NUM_PROMPTS -> $RESULT_DIR/${NAME}.*"

# --warmup-requests 0: in multi-turn mode a warmup request replays a WHOLE
# conversation, which pre-warms the very cache this run is measuring.
python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url "$ROUTER_URL" \
  --model "$TOKENIZER" --tokenizer "$TOKENIZER" \
  --served-model-name "$SERVED_MODEL" \
  --dataset-name agentic-trace --dataset-path "$TRACE" \
  --sharegpt-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
  --warmup-requests "${WARMUP:-0}" \
  --cache-report --output-details --output-file "$DETAILS" \
  ${EXTRA_BENCH_ARGS:-} 2>&1 | tee "$LOG_DIR/${NAME}.log"

# The tool's own input and cache summary is wrong in multi-turn mode -- it keeps
# the conversation-level prompt_len for every turn -- so this is the number to
# read. See README §5.3.
echo
python3 "$HERE/score_agentic_trace.py" "$TRACE" "$DETAILS" "$NUM_PROMPTS" \
  --page-size "$PAGE_SIZE" | tee "$RESULT_DIR/${NAME}.score.txt"
