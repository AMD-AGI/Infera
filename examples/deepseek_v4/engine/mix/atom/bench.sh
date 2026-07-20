#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: throughput sweep against the router using atom's benchmark_serving (openai backend).
# why : optional perf measurement for external users. how: sweep concs from argv; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL
CTR="${CTR:-dsv4_atom_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"; CONCS="${*:-64}"; OUT="${OUT:-/tmp/mix_atom_out}"
docker exec "$CTR" mkdir -p "$OUT"
for C in $CONCS; do
  docker exec "$CTR" bash -lc "python3 -m atom.benchmarks.benchmark_serving --backend openai \
    --base-url http://$NODE_IP:$ROUTER_PORT --model $INFERA_MODEL --tokenizer $INFERA_MODEL \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
    --max-concurrency $C --num-prompts $((C*10)) --num-warmups $((C*2)) --request-rate inf --ignore-eos \
    --save-result --result-dir $OUT --result-filename atom_c${C}.json 2>&1 | tail -20"
done
log "bench done -> in-container $OUT/*.json"
