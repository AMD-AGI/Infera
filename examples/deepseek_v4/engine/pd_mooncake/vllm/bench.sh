#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: PD throughput sweep against the router using vllm bench serve (completions path).
# why : optional perf run for external users. how: sweep concs from argv; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env PREFILL_NODE; require_env PREFILL_IP
CTR="${CTR:-dsv4_pd_vllm}"; ISL="${ISL:-8192}"; OSL="${OSL:-1024}"; CONCS="${*:-32}"; OUT="${OUT:-/tmp/pd_vllm_out}"
for C in $CONCS; do
  ssh -o StrictHostKeyChecking=no "$PREFILL_NODE" "docker exec $CTR bash -lc 'mkdir -p $OUT; \
    vllm bench serve --backend vllm --model $INFERA_MODEL --endpoint /v1/completions \
    --base-url http://$PREFILL_IP:$ROUTER_PORT --dataset-name random --random-input-len $ISL \
    --random-output-len $OSL --num-prompts $((C*10)) --max-concurrency $C --ignore-eos --save-result \
    --result-dir $OUT --result-filename vllm_pd_c${C}.json 2>&1 | tail -20'"
done
log "bench done -> $PREFILL_NODE:$OUT/*.json"
