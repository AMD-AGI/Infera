#!/usr/bin/env bash
# conc stress against the PD router. Long ISL on purpose: at per-rank chunk 2048 an ISL=32768
# request is ~16 prefill chunks, i.e. exactly the multi-chunk path the wait_event fix targets.
# Run INSIDE the prefill container (pd_uni), pointed at the ROUTER (never a leg directly).
set -uo pipefail
NAME="${NAME:-pd_uni}"
BASE_HOST="${BASE_HOST:-10.2.122.44}"
PORT="${PORT:-8002}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CONC="${CONC:-32}"
NUM="${NUM:-$((CONC * 2))}"
ISL="${ISL:-32768}"
OSL="${OSL:-256}"
TAG="${TAG:-patched}"
OUT="${OUT:-/tmp/bench_${TAG}_c${CONC}_isl${ISL}.jsonl}"
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"

docker exec "$NAME" python3 -m sglang.bench_serving \
  --backend sglang-oai --host "$BASE_HOST" --port "$PORT" --model "$SERVED" \
  --tokenizer "$TOKENIZER" \
  --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio 1.0 \
  --num-prompts "$NUM" --max-concurrency "$CONC" \
  --output-file "$OUT" 2>&1 | grep -avE "^Namespace|benchmark_args|^WARNING|Fail to load tokenizer"
