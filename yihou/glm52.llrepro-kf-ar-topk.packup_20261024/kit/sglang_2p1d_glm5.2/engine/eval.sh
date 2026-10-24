#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# GSM8K the way InferenceX runs it: benchmark_lib.sh's run_eval at the pinned
# InferenceX commit (its lm-eval revision and patches, utils/evals/gsm8k.yaml,
# 5-shot chat completions with the chat template, greedy, the full 1,319-question
# test set) through the router, from a throwaway client container of the same
# image. Run it on a deployment with real MTP acceptance. lm-eval installs on
# each run; the dataset is cached in $WORK_DIR/cache/gsm8k. Output: <run>/gsm8k/.
#
# Usage: bash engine/eval.sh     (EVAL_CONC requests in flight; EVAL_LIMIT=<n> for a slice)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
RUN_DIR="$(run_dir)"
inferencex_checkout

out="$RUN_DIR/gsm8k"
cache="$WORK_DIR/cache/gsm8k"
mkdir -p "$out" "$cache"
# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --name "$CTR-eval" --network host \
    -v "$IX:$IX:ro" -v "$out:$out" -v "$cache:/cache" -v "$MODEL:$MODEL:ro" \
    -e "MODEL=$MODEL" -e "MODEL_PATH=$MODEL" -e "MODEL_NAME=$SERVED_MODEL" -e "IMAGE=$IMAGE" \
    -e "EVAL_CONCURRENT_REQUESTS=$EVAL_CONC" -e "EVAL_LIMIT=${EVAL_LIMIT:-}" -e HF_HOME=/cache/hf \
    "$IMAGE" bash -c '
        ix="$1" out="$2" port="$3" owner="$4"
        cd "$ix"
        source benchmarks/benchmark_lib.sh
        run_eval --port "$port" --results-dir "$out"
        rc=$?
        chown -R "$owner" "$out" /cache
        exit "$rc"
    ' _ "$IX" "$out" "$ROUTER_PORT" "$(id -u):$(id -g)" 2>&1 | tee "$out/eval.log"
