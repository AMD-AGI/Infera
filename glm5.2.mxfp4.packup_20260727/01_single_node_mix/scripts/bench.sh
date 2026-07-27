#!/usr/bin/env bash
# conc=64 stress test, 1k/1k profile (user-chosen fast validation). Runs INSIDE the container.
set -uo pipefail
NAME="${NAME:-glm52-mix-p1}"
BASE="${BASE:-http://127.0.0.1:30000}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
CONC="${CONC:-64}"; ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
NP="${NP:-$((CONC*4))}"   # 256 prompts @ conc64 — enough to saturate, quick
docker exec "$NAME" bash -lc "python3 -m sglang.bench_serving --backend sglang-oai \
  --base-url $BASE --model $MODEL --tokenizer $MODEL \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
  --max-concurrency $CONC --num-prompts $NP --warmup-requests $((CONC)) \
  --request-rate inf 2>&1 | tail -35"
