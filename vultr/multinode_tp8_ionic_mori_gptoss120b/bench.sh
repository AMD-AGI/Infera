#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Throughput bench against the 1P1D router. Runs from inside the prefill
# container (localhost router). Sweeps concurrency; ISL/OSL fixed.
set -euo pipefail
CONTAINER="${CONTAINER:-pd_mori_sgl}"
ROUTER_HOST="${ROUTER_HOST:?set ROUTER_HOST}"; ROUTER_PORT="${ROUTER_PORT:-8000}"
MODEL="${MODEL:-/PATH/TO/gpt-oss-120b}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
SWEEP="${SWEEP:-16 32 64 128 256}"
TAG="${TAG:-pd_mori_1p1d_gptoss}"
RANGE_RATIO="${RANGE_RATIO:-1.0}"
PROMPTS_MULT="${PROMPTS_MULT:-10}"; PROMPTS_MAX="${PROMPTS_MAX:-8192}"
WALL="${WALL:-600}"
BENCH="${BENCH:-/sgl-workspace/sglang/python/sglang/bench_serving.py}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/results/${TAG}}"
mkdir -p "$OUTDIR"
for C in $SWEEP; do
    NP=$(( C * PROMPTS_MULT )); (( NP > PROMPTS_MAX )) && NP=$PROMPTS_MAX
    OUT="${OUTDIR}/${TAG}_conc${C}_isl${ISL}_osl${OSL}.json"
    echo "===== CONC=$C ISL=$ISL OSL=$OSL prompts=$NP ====="
    ssh "${PREFILL_NODE:?set PREFILL_NODE}" "docker exec -e HF_HUB_OFFLINE=0 ${CONTAINER} bash -lc '
        timeout --foreground --kill-after=10s ${WALL}s python3 ${BENCH} \
            --backend sglang-oai --base-url http://${ROUTER_HOST}:${ROUTER_PORT} \
            --model ${MODEL} --dataset-name random \
            --random-input-len ${ISL} --random-output-len ${OSL} --random-range-ratio ${RANGE_RATIO} \
            --num-prompts ${NP} --max-concurrency ${C} --request-rate inf \
            --warmup-requests ${C} --output-file ${OUT}' 2>&1" \
        | grep -vE 'it/s\]|Namespace|Warning' | tail -25
done
echo "results in $OUTDIR"
