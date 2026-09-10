#!/bin/bash
set -euo pipefail
W=/shared_nfs/yihou/playground/glm52_decode_fake_20260909_1016
ROUND=${1:?round name}
N=${2:-128}
R="$W/rounds/$ROUND"
mkdir -p "$R"
CMD=(python3 -m sglang.benchmark.serving
    --backend sglang --host 127.0.0.1 --port 31816
    --model /shared_nfs/models/GLM-5.2-MXFP4
    --dataset-name random --dataset-path "$W/reference/synthetic_prompts.json"
    --tokenize-prompt --random-input-len 289652 --random-output-len 774
    --random-range-ratio 1 --num-prompts "$N" --max-concurrency 16
    --warmup-requests 16 --fake-prefill --output-details
    --output-file "$R/benchmark.jsonl")
printf '%q ' "${CMD[@]}" > "$R/benchmark-command.txt"; printf '\n' >> "$R/benchmark-command.txt"
docker exec -e PYTHONPATH=/sglang/python -e PYTHONDONTWRITEBYTECODE=1 g52-faked-1016 "${CMD[@]}"
