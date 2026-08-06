#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: a REFERENCE throughput sweep against the router, using sglang's own bench_serving.
# why : it ships inside the engine image, so there is nothing extra to install, and it is
#       enough to confirm the deployment performs sanely. It is NOT the agentic benchmark —
#       see results/ for what the agentic numbers look like and where those harnesses live.
# how : bash cluster/<your-cluster>.sh bench [conc ...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env PREFILL_IP; require_env MODEL
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
URL="http://$PREFILL_IP:$ROUTER_PORT"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONCS="${*:-8}"
OUT="${OUT:-/tmp/glm52_bench}"

for C in $CONCS; do
  N="${N:-$((C * 10))}"
  WARM=$([ "$C" -lt 8 ] && echo "$C" || echo 8)
  TAG="glm52_isl${ISL}_osl${OSL}_c${C}"
  log "=== $TAG ==="
  # --random-range-ratio 1.0 pins every prompt to exactly ISL. A fixed-length sweep wants a
  # delta, not a distribution; the default draws uniformly and the percentiles then mix
  # request sizes.
  #
  # Sampling uses the model's own generation_config rather than greedy. temperature 0 sends
  # this reasoning model into repetition on a long prompt, MTP then predicts the loop
  # perfectly, acceptance length pins at 4.00, and the run reads like KV corruption.
  #
  # --cache-report needs the server's --enable-cache-report (engine/leg.sh passes it). Note
  # what the column means for THIS dataset: --dataset-name random builds every prompt
  # independently, so there is no shared prefix by construction and any nonzero cache hit is
  # residue from the previous round, not a cache result. Prefix reuse is an agentic-workload
  # property; see results/.
  $SSH_CMD "$PREFILL_NODE" "docker exec $CTR bash -c 'mkdir -p $OUT && \
    python3 -m sglang.bench_serving \
      --backend sglang-oai-chat --base-url $URL \
      --model $SERVED --tokenizer ${TOKENIZER:-$MODEL} \
      --dataset-name random --random-input-len $ISL --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --max-concurrency $C --num-prompts $N --request-rate inf \
      --warmup-requests $WARM \
      --cache-report --temperature 1.0 --top-p 0.95 \
      --output-file $OUT/$TAG.jsonl --output-details 2>&1 | tail -30'"
done

log "bench done -> $PREFILL_NODE:$OUT/*.jsonl (per-request ttfts/itls are in the jsonl)"
