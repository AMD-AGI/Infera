#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# GSM8K accuracy through the router: lm_eval 0.4.13 (installed into a throwaway
# client container of the same image) with 5-shot multi-turn chat, greedy
# decoding. Run it on an engine started with real MTP acceptance. The package
# and the dataset download on each run (pip and Hugging Face caches persist in
# $WORK_DIR/cache/gsm8k). Output: <run>/gsm8k/.
#
# Usage: bash engine/eval.sh     (EVAL_LIMIT=200 questions, EVAL_CONCURRENCY=64 requests)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
RUN_DIR="$(run_dir)"

out="$RUN_DIR/gsm8k"
cache="$WORK_DIR/cache/gsm8k"
mkdir -p "$out" "$cache"
model_args="model=$SERVED_MODEL,base_url=$ROUTER_URL/v1/chat/completions,tokenizer=$MODEL"
model_args+=",tokenized_requests=False,num_concurrent=${EVAL_CONCURRENCY:-64}"
model_args+=",max_gen_toks=16384,max_length=1048576,timeout=900,trust_remote_code=True"

# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --name "$CTR-eval" --network host \
    -v "$out:$out" -v "$cache:/cache" -v "$MODEL:$MODEL:ro" \
    -e HF_HOME=/cache/hf -e PIP_CACHE_DIR=/cache/pip "$IMAGE" bash -c '
        out="$1" owner="$2" model_args="$3" limit="$4"
        pip install -q "lm-eval[api]==0.4.13" &&
            python3 -m lm_eval --model local-chat-completions --model_args "$model_args" \
                --tasks gsm8k --num_fewshot 5 --limit "$limit" \
                --apply_chat_template --fewshot_as_multiturn \
                --gen_kwargs temperature=0,top_p=1,max_gen_toks=16384 \
                --log_samples --output_path "$out"
        rc=$?
        chown -R "$owner" "$out" /cache
        exit "$rc"
    ' _ "$out" "$(id -u):$(id -g)" "$model_args" "${EVAL_LIMIT:-200}" 2>&1 | tee "$out/eval.log"
