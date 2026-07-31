#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Concurrency sweep against the server run_sglang.sh started, via sglang.bench_serving
# (already in the image). Results land in bench_<TAG>/ next to this script.
# Override example: MODEL=... ISL=65536 OSL=512 CONC="8 16 32" TAG=longctx bash bench.sh
set -euo pipefail

: "${MODEL:?set MODEL=/path/to/GLM-5.2-FP8 (used for the tokenizer as well)}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-16 64 128 256}"
RANGE="${RANGE:-0.8}"    # align with InferenceX recipes; len in [RANGE*len, len]. 
TAG="${TAG:-agg}"
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench_${TAG}}"
mkdir -p "${OUT}"

for c in ${CONC}; do
    name="${TAG}_isl${ISL}_osl${OSL}_c${c}"
    echo "[bench] ${name} -> ${OUT}/${name}.json"
    python3 -m sglang.bench_serving \
        --backend sglang-oai --host "${HOST}" --port "${PORT}" \
        --model "${MODEL}" --tokenizer "${MODEL}" \
        --dataset-name random --random-input-len "${ISL}" --random-output-len "${OSL}" \
        --random-range-ratio "${RANGE}" \
        --num-prompts "$((c * 10))" --max-concurrency "${c}" --request-rate inf \
        --warmup-requests "$((c * 2))" \
        --output-file "${OUT}/${name}.json" 2>&1 | tee "${OUT}/${name}.log"
done

echo "[bench] results in ${OUT}"
