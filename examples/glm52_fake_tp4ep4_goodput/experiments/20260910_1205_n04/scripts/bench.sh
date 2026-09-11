#!/bin/bash
set -euo pipefail
W=/apps/tas/yaoc/research/topic/infera-with-hyperloom/exp_only/Infera-yx-test/examples/glm52_fake_tp4ep4_goodput/experiments/20260910_1205_n04
ROUND=${1:?round name}
N=${2:-128}
CONC=${3:-16}
R="$W/rounds/$ROUND"
mkdir -p "$R"
CMD=(python3 -m sglang.benchmark.serving
    --backend sglang --host 127.0.0.1 --port 31864
    --model /shared_nfs/models/GLM-5.2-MXFP4
    --dataset-name random --dataset-path "$W/reference/synthetic_prompts.json"
    --tokenize-prompt --random-input-len 10000 --random-output-len 500
    --random-range-ratio 1 --num-prompts "$N" --max-concurrency "$CONC"
    --warmup-requests 16 --fake-prefill --output-details
    --output-file "$R/benchmark.jsonl")
printf '%q ' "${CMD[@]}" > "$R/benchmark-command.txt"; printf '\n' >> "$R/benchmark-command.txt"
docker exec -e PYTHONPATH=/sglang/python -e PYTHONDONTWRITEBYTECODE=1 glm52-goodput-20260910-1205 "${CMD[@]}"
