#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: throughput sweep against the router using vllm bench serve (openai-compatible).
# why : optional perf measurement for external users. how: sweep concs from argv; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL
CTR="${CTR:-dsv4_vllm_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"; CONCS="${*:-256}"; OUT="${OUT:-/tmp/mix_vllm_out}"
docker exec "$CTR" mkdir -p "$OUT"
for C in $CONCS; do
  docker exec "$CTR" bash -lc "vllm bench serve --backend vllm --model $INFERA_MODEL \
    --endpoint /v1/completions --base-url http://$NODE_IP:$ROUTER_PORT \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL \
    --num-prompts $((C*10)) --max-concurrency $C --ignore-eos --save-result \
    --result-dir $OUT --result-filename vllm_c${C}.json 2>&1 | tail -20"
done
log "bench done -> in-container $OUT/*.json"
