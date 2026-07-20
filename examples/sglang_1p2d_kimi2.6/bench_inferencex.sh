#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# InferenceX serving bench (utils/bench_serving/benchmark_serving.py) against the router (127.0.0.1:8000).
# benchmark_serving.py is NOT baked into the image; it is mounted from the host, so IX must point to a
# local clone of the InferenceX repo. Richer metrics than `vllm bench serve` (TTFT/ITL/TPOT/E2E percentiles).
# Override: IX=/path/to/InferenceX ISL=1024 OSL=1024 CONC=512 RATE=inf bash bench_inferencex.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (local model dir, bind-mounted read-only)}"
: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
: "${IX:?set IX=/path/to/InferenceX (local InferenceX clone; benchmark_serving.py is mounted from here)}"
BACKEND="${BACKEND:-openai}"                   # openai=/v1/completions, openai-chat=/v1/chat/completions
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC="${CONC:-512}"
RATE="${RATE:-inf}"                            # req/s; inf = fire all (keeps the pipe full)
RANGE="${RANGE:-0.8}"                          # align with InferenceX recipes (deepseek-v4 disagg=0.8). len ∈ [RANGE*OSL, OSL]
NUM_PROMPTS="${NUM_PROMPTS:-$((CONC * 10))}"
OUT="${OUT:-$(cd "$(dirname "$0")" && pwd)}"
TAG="ix_isl${ISL}_osl${OSL}_c${CONC}_r${RATE}"

[[ -f "$IX/utils/bench_serving/benchmark_serving.py" ]] || { echo "[ix-bench] not found: $IX/utils/bench_serving/benchmark_serving.py — set IX=<InferenceX clone>"; exit 1; }

echo "[ix-bench] $HOST:$PORT backend=$BACKEND ISL=$ISL OSL=$OSL CONC=$CONC RATE=$RATE range=$RANGE prompts=$NUM_PROMPTS IX=$IX -> $OUT/$TAG.{json,log}"
docker run --rm --network host -v "$IX:$IX" -v "$MODEL:$MODEL:ro" -v "$OUT:$OUT" "$IMAGE" \
    bash -c "cd '$IX/utils/bench_serving' && python benchmark_serving.py \
        --backend '$BACKEND' --host '$HOST' --port '$PORT' \
        --model '$MODEL' --tokenizer '$MODEL' --trust-remote-code \
        --dataset-name random \
        --random-input-len '$ISL' --random-output-len '$OSL' --random-range-ratio '$RANGE' \
        --num-prompts '$NUM_PROMPTS' --max-concurrency '$CONC' --request-rate '$RATE' \
        --save-result --result-dir '$OUT' --result-filename '$TAG.json'" \
    2>&1 | tee "$OUT/$TAG.log"
