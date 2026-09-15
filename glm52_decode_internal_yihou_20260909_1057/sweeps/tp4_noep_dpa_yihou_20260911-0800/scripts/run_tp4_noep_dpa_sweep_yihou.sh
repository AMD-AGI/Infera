#!/usr/bin/env bash
# TP4 / EP1 (no expert parallel) / DP-attention sweep over global concurrency.
# Runs only on the existing allocation and container supplied by the caller.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${JOB_ID:?}" "${NODE:?}" "${CONTAINER:?}" "${OUTPUT_ROOT:?Set a unique sweep iterations directory}"
POINTS=("$@")
[[ ${#POINTS[@]} -gt 0 ]] || POINTS=(4 8 16 20 24)
for concurrency in "${POINTS[@]}"; do
    args=(--tp-size 4 --ep-size 1 --enable-dp-attention --batch-size "$concurrency"
        --input-len 70000 --output-len 10000 --accept-length 3.61
        --warmup-steps 10 --enable-aiter-allreduce-fusion
        --enable-fused-qk-norm-rope --mem-fraction-static 0.85)
    bash "$ROOT/scripts/run_decode.sh" "noep_dpa_on_c${concurrency}_yihou" "${args[@]}"
done
