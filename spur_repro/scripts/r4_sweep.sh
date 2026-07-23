#!/usr/bin/env bash
set -u
CONCS="${1:-2 32}"; PORT="${2:-30000}"; OUTDIR="${3:-/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round2_bench/out}"; TAG="${4:-nodp}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"
mkdir -p "$OUTDIR"
for C in $CONCS; do
  NP=$((C * 10)); WU=$((C * 2))
  OUT="$OUTDIR/r4_${TAG}_c${C}.jsonl"
  echo "=== [$TAG] conc=$C num_prompts=$NP warmup=$WU -> $OUT ==="
  python3 -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$C" --num-prompts "$NP" \
    --warmup-requests "$WU" --request-rate inf \
    --output-file "$OUT" 2>&1 | tail -30
  echo "=== conc=$C done ==="
done
