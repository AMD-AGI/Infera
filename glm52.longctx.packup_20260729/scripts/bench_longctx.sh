#!/usr/bin/env bash
# conc=32 stress on the long-context DPA server. Uses sglang.bench_serving random dataset.
# ISL is intentionally long (default 32K) so this stresses the SAME long-prefill path the
# needle probe validated, not a 1k/1k toy load.
set -uo pipefail
NAME="${NAME:-glm52-dpa-longctx}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CONC="${CONC:-32}"
NUM="${NUM:-$((CONC * 2))}"
ISL="${ISL:-32768}"
OSL="${OSL:-256}"
OUT="${OUT:-/tmp/bench_c${CONC}_isl${ISL}.jsonl}"
# served-model-name is not a local dir; bench_serving needs the real checkpoint for the tokenizer
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"

docker exec "$NAME" python3 -m sglang.bench_serving \
  --backend sglang-oai --host "$HOST" --port "$PORT" --model "$SERVED" \
  --tokenizer "$TOKENIZER" \
  --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio 1.0 \
  --num-prompts "$NUM" --max-concurrency "$CONC" \
  --output-file "$OUT"
