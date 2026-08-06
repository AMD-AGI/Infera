#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Simple serving benchmark through the router, using SGLang's own bench_serving on
# a random-token dataset. Run INSIDE the engine container on the PREFILL node:
#   bash bench.sh
#   ISL=8192 OSL=512 CONC=32 bash bench.sh
#
# This sizes the deployment; it is not the agentic workload the tuned recipe was
# chosen on. Random prompts share no prefix, so the kv-aware router has nothing to
# reuse and the cache-hit line below reads ~0 by construction -- that is correct,
# not a fault. See README §5.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

TAG="${TAG:-random_isl${ISL}_osl${OSL}_c${CONC}_n${NUM_PROMPTS}}"
OUT="$RESULT_DIR/$TAG"

curl -sf -m 10 "$ROUTER_URL/health" >/dev/null \
  || { echo "[bench] router is not answering at $ROUTER_URL -- run verify.sh first" >&2; exit 1; }

echo "[bench] $ROUTER_URL ISL=$ISL OSL=$OSL CONC=$CONC prompts=$NUM_PROMPTS -> $OUT.{json,log}"

# --warmup-requests 1 keeps CUDA-graph capture and the first-token path out of the
# measured window. It cannot pre-warm a prefix here because the prompts are random.
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --host "$PREFILL_IP" --port "$ROUTER_PORT" \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio "${RANGE:-0.8}" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
  --request-rate "${RATE:-inf}" --warmup-requests "${WARMUP:-1}" \
  --seed "${SEED:-42}" \
  --cache-report --output-details --output-file "$OUT.json" \
  ${EXTRA_BENCH_ARGS:-} 2>&1 | tee "$OUT.log"

# The offload tier's counters are the only place its side of the run is recorded,
# and they are cumulative, so capture them next to the result rather than leaving
# them to be read later against a different total. See README §6.
if [[ "$KVD" == "1" ]]; then
  python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" > "$OUT.kvd.json" 2>/dev/null \
    && echo "[bench] kvd counters: $OUT.kvd.json" \
    || echo "[bench] kvd counters unavailable on $KVD_SOCKET" >&2
fi

echo "[bench] done: $OUT.json"
