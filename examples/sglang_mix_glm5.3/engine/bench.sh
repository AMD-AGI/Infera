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
