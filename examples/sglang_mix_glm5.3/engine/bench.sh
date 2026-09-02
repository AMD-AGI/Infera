#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Reference fixed-length sweep using sglang's own bench_serving, which ships
# inside the image.
#
# Use this rather than a shell fan-out of curl. A bash loop of N concurrent
# curls becomes the bottleneck well before the engine does: at concurrency 32
# one measured 350 output tok/s while the engine's own log reported 2398 tok/s
# at #running-req 32 with an empty queue. The client was the limit, not the server.
#
# Three flags are load-bearing and are NOT defaults:
#   --random-range-ratio 1.0   pins every prompt to exactly ISL. The default
#                              draws uniformly and the percentiles then mix
#                              request sizes; a fixed-length sweep wants a
#                              delta, not a distribution.
#   --temperature 1.0 --top-p 0.95   the checkpoint's own generation_config
#                              defaults, deliberately NOT greedy. At temperature
#                              0 this reasoning model falls into repetition on a
#                              long prompt and the run reads like corruption.
#   --num-prompts 10 x conc    enough requests per arm to reach steady state.
#
# KNOWN LIMITATION — long OSL produces degenerate output. Measured, not feared.
#
# The default OSL of 320 is clean. Raise it and the generations collapse into
# repetition loops, at a rate that grows with length. Measured on this stack with
# a 10-gram-repeated->=5x check, on both GLM-5.3 and GLM-5.2:
#
#     osl   320   0-1 %  of requests
#     osl  3300    40-65 %
#     osl 17000    96-100 %      (worst: one 10-gram repeated 16,990 times)
#
# The tok/s arithmetic is NOT affected -- degenerate requests were measured to
# decode at the same speed as clean ones (+0.2 % at concurrency 1). What is wrong
# is *what the tokens are*. So a long-OSL number here is a valid throughput
# measurement of the engine generating repetitive text, and must not be quoted as
# a quality result.
#
# Cause is NOT established. The leading candidate is this script's own shape:
# `--dataset-name random` with no `--apply-chat-template` sends a bag of random
# tokens with no conversational framing and asks the model to continue it. Prior
# experience on GLM-5.2 is that applying the chat template plus the checkpoint's
# official temperature/top-p mitigates it. A second candidate is the ROCm silent
# greedy fallback in EAGLE verify (`eagle_utils.py:726`, `_is_hip` sits in an
# `or` with `is_all_greedy`, so temperature and top_p never reach token selection
# -- three open upstream PRs, #31214 / #32922 / #37134, none merged). The two are
# not mutually exclusive and both were present in every arm measured.
#
# If you need long-OSL numbers you can defend on quality, add
# `--apply-chat-template` and score the saved generations for n-gram repetition
# (`--output-details` saves `generated_texts`) rather than trusting the summary.
# ACCEPTANCE LENGTH DOES NOT DETECT THIS: an aggregate of 2.96 with 1.25 % at the
# 4.00 ceiling was measured while 54 % of the very same requests were looping.
#
# Usage: bash engine/bench.sh [conc ...]      (default 1 8 16 24)
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
SERVED="${SERVED:-glm5.3-$VARIANT}"
ISL="${ISL:-7400}"; OSL="${OSL:-320}"
CONCS=("$@"); [ ${#CONCS[@]} -eq 0 ] && CONCS=(1 8 16 24)

for c in "${CONCS[@]}"; do
  echo "===== isl=$ISL osl=$OSL conc=$c ====="
  docker exec "$CTR" python3 -m sglang.bench_serving --backend sglang-oai-chat \
    --host "$MY_IP" --port "$PORT" --model "$SERVED" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 --max-concurrency "$c" --num-prompts $((c * 10)) \
    --temperature 1.0 --top-p 0.95 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Median TTFT|P99 TTFT|Mean TPOT|Median E2E"
done
