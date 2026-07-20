#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: PD throughput sweep against the router using atom benchmark_serving (completions path).
# why : optional perf run for external users. how: sweep concs from argv; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env PREFILL_NODE; require_env PREFILL_IP
CTR="${CTR:-dsv4_pd_atom}"; ISL="${ISL:-8192}"; OSL="${OSL:-1024}"; CONCS="${*:-64}"; OUT="${OUT:-/tmp/pd_atom_out}"
for C in $CONCS; do
  ssh -o StrictHostKeyChecking=no "$PREFILL_NODE" "docker exec $CTR bash -lc 'mkdir -p $OUT; \
    python3 -m atom.benchmarks.benchmark_serving --backend openai --base-url http://$PREFILL_IP:$ROUTER_PORT \
    --model $INFERA_MODEL --tokenizer $INFERA_MODEL --dataset-name random --random-input-len $ISL \
    --random-output-len $OSL --random-range-ratio 1.0 --max-concurrency $C --num-prompts $((C*10)) \
    --num-warmups $((C*2)) --request-rate inf --ignore-eos --save-result --result-dir $OUT \
    --result-filename atom_pd_c${C}.json 2>&1 | tail -20'"
done
log "bench done -> $PREFILL_NODE:$OUT/*.json"
