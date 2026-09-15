#!/bin/bash
set -euo pipefail
W=/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/glm52_decode_internal_yihou_20260909_1057/comparisons/reverse_fake_server_sweep_yihou_20260910-0837
ROUND=${1:?round name}
N=${2:-128}
CONC=${3:-16}
R="$W/rounds/$ROUND"
mkdir -p "$R"
CMD=(python3 -m sglang.benchmark.serving
    --backend sglang --host 127.0.0.1 --port 31916
    --model /shared_nfs/models/GLM-5.2-MXFP4
    --dataset-name random --dataset-path "$W/reference/synthetic_prompts.json"
    --tokenize-prompt --random-input-len 70000 --random-output-len 10000
    --random-range-ratio 1 --num-prompts "$N" --max-concurrency "$CONC"
    --warmup-requests 16 --fake-prefill --output-details
    --output-file "$R/benchmark.jsonl")
printf '%q ' "${CMD[@]}" > "$R/benchmark-command.txt"; printf '\n' >> "$R/benchmark-command.txt"
docker exec -e PYTHONPATH=/sglang/python -e PYTHONDONTWRITEBYTECODE=1 "${NAME:-yihou-glm52-reverse-0910-off-c16}" "${CMD[@]}"
