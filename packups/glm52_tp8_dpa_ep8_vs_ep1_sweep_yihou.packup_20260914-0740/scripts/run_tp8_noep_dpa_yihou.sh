#!/usr/bin/env bash
# Phase 3: the SAME sweep with expert parallelism OFF (--ep-size 1), i.e. a plain TP MoE.
# Byte-for-byte identical to run_tp8_ep8_dpa_yihou.sh except `--ep-size 1` and the iteration name,
# so that EP8-vs-EP1 is a single-variable comparison. Run it in the SAME container as the EP8
# sweep — that is the whole point of the control.
set -euo pipefail
WS=$(cd "$(dirname "$0")/.." && pwd)
for concurrency in "${@:?usage: run_tp8_noep_dpa_yihou.sh C [C ...]}"; do
    bash "$WS/scripts/run_decode.sh" "tp8_noep_dpa_on_c${concurrency}_yihou" \
        --tp-size 8 --ep-size 1 --enable-dp-attention --batch-size "$concurrency" \
        --max-running-requests "$concurrency" \
        --input-len 70000 --output-len 10000 --accept-length 3.61 \
        --warmup-steps 10 --enable-aiter-allreduce-fusion \
        --enable-fused-qk-norm-rope --mem-fraction-static 0.85
done
