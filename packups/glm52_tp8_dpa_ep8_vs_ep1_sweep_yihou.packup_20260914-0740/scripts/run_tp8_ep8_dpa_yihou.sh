#!/usr/bin/env bash
# The TP8 / EP8 / DP-attention decode point(s). Same parameters as the packup's EP4 control except
# tp=8, ep=8 and the concurrency list. Default: C=128.
set -euo pipefail
WS=$(cd "$(dirname "$0")/.." && pwd)
for concurrency in "${@:-128}"; do
    bash "$WS/scripts/run_decode.sh" "tp8_ep8_dpa_on_c${concurrency}_yihou" \
        --tp-size 8 --ep-size 8 --enable-dp-attention --batch-size "$concurrency" \
        --max-running-requests "$concurrency" \
        --input-len 70000 --output-len 10000 --accept-length 3.61 \
        --warmup-steps 10 --enable-aiter-allreduce-fusion \
        --enable-fused-qk-norm-rope --mem-fraction-static 0.85
done
