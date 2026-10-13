#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# GSM8K accuracy through the router: lm_eval (shipped in the ATOM nightly base)
# with 5-shot multi-turn chat, greedy decoding. Needs real MTP acceptance.
# The dataset downloads from the Hugging Face Hub on first use.
# Output: <run>/gsm8k/.
#
# Usage: bash engine/eval.sh     (EVAL_LIMIT=200 questions, EVAL_CONCURRENCY=64 requests)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
load_run
require_real_acceptance

out="$RUN_DIR/gsm8k"
cache="$WORK_DIR/cache/gsm8k"
mkdir -p "$out" "$cache"
model_args="model=$MODEL,base_url=$ROUTER_URL/v1/chat/completions,tokenizer=$MODEL"
model_args+=",tokenized_requests=False,num_concurrent=${EVAL_CONCURRENCY:-64}"
model_args+=",max_gen_toks=16384,max_length=1048576,timeout=900,trust_remote_code=True"

docker run --rm --name "$CTR-eval" --network host --user "$(id -u):$(id -g)" \
    -v "$WORK_DIR:$WORK_DIR" -v "$MODEL:$MODEL:ro" \
    -e HF_HOME="$cache" -e XDG_CACHE_HOME="$cache/xdg" "$IMAGE" \
    python3 -m lm_eval --model local-chat-completions --model_args "$model_args" \
    --tasks gsm8k --num_fewshot 5 --limit "${EVAL_LIMIT:-200}" \
    --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs temperature=0,top_p=1,max_gen_toks=16384 \
    --log_samples --output_path "$out" 2>&1 | tee "$out/eval.log"
