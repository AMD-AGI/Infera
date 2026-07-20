#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: PD throughput sweep against the router (sglang-oai backend).
# why : optional perf run for external users. how: sweep concs from argv; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env PREFILL_NODE; require_env PREFILL_IP
CTR="${CTR:-dsv4_pd_sgl_mori}"; ISL="${ISL:-8192}"; OSL="${OSL:-1024}"; CONCS="${*:-640}"; OUT="${OUT:-/tmp/pd_sgl_out}"
for C in $CONCS; do
  ssh -o StrictHostKeyChecking=no "$PREFILL_NODE" "docker exec $CTR bash -lc 'mkdir -p $OUT; \
    python3 -m sglang.bench_serving --backend sglang-oai --base-url http://$PREFILL_IP:$ROUTER_PORT \
    --model $INFERA_MODEL --tokenizer $INFERA_MODEL --dataset-name random --random-input-len $ISL \
    --random-output-len $OSL --random-range-ratio 1.0 --max-concurrency $C --num-prompts $((C*10)) \
    --warmup-requests $((C*2)) --request-rate inf --output-file $OUT/pd_c${C}.jsonl 2>&1 | tail -20'"
done
log "bench done -> $PREFILL_NODE:$OUT/*.jsonl"
