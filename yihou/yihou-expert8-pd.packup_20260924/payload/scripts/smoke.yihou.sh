#!/usr/bin/env bash
# Operational smoke only: no factual/tool-call quality assertions for synthetic weights.
set -euo pipefail
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
ROUND="${1:-$W/rounds/002-bringup}"
bash "$W/scripts/bench_fixed.yihou.sh" "$ROUND/smoke-single" \
    --input-len 128 --output-len 16 --concurrency 1 --warmup 0 --requests 1
bash "$W/scripts/bench_fixed.yihou.sh" "$ROUND/smoke-burst" \
    --input-len 256 --output-len 32 --concurrency 8 --warmup 0 --requests 16
