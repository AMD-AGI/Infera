#!/usr/bin/env bash
# Same-node EP4 control points, to separate the EP effect from node-to-node variation.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${JOB_ID:?}" "${NODE:?}" "${CONTAINER:?}" "${OUTPUT_ROOT:?}"
for concurrency in "$@"; do
    bash "$ROOT/scripts/run_decode.sh" "control_ep4_dpa_on_c${concurrency}_yihou" \
        --tp-size 4 --ep-size 4 --enable-dp-attention --batch-size "$concurrency" \
        --input-len 70000 --output-len 10000 --accept-length 3.61 \
        --warmup-steps 10 --enable-aiter-allreduce-fusion \
        --enable-fused-qk-norm-rope --mem-fraction-static 0.85
done
