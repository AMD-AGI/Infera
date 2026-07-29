#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Throughput sweep through the router, using sglang.bench_serving (already in the
# image). Run on the prefill node once curl.sh passes.
# Override: ISL=8192 OSL=1024 CONC="64 128 256" bash bench.sh
# For the aggregated baseline: PORT=30000 TAG=agg bash bench.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64 128 256 384 512}"
TAG="${TAG:-pd}"
OUT="${OUT:-$(cd "$(dirname "$0")" && pwd)/bench_${TAG}}"
mkdir -p "$OUT"

for c in $CONC; do
    name="${TAG}_isl${ISL}_osl${OSL}_c${c}"
    echo "[bench] $name -> $OUT/$name.json"
    python3 -m sglang.bench_serving \
        --backend sglang-oai --host "$HOST" --port "$PORT" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
        --random-range-ratio 1.0 \
        --num-prompts "$((c * 5))" --max-concurrency "$c" --request-rate inf \
        --output-file "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
done

echo "[bench] results in $OUT"
