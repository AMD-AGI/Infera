#!/usr/bin/env bash
# Run only on the existing allocation/container supplied by the caller.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${JOB_ID:?}" "${NODE:?}" "${CONTAINER:?}" "${OUTPUT_ROOT:?Set a unique sweep iterations directory}"
for mode in off on; do
    for concurrency in 4 8 16 20 24; do
        args=(--tp-size 4 --ep-size 4 --batch-size "$concurrency"
            --input-len 70000 --output-len 10000 --accept-length 3.61
            --warmup-steps 10 --enable-aiter-allreduce-fusion
            --enable-fused-qk-norm-rope --mem-fraction-static 0.85)
        [[ "$mode" == off ]] || args+=(--enable-dp-attention)
        bash "$ROOT/scripts/run_decode.sh" "full_dpa_${mode}_c${concurrency}_yihou" "${args[@]}"
    done
done
