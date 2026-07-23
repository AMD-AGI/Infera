#!/usr/bin/env bash
# PD sweep against the mini-lb router (:8100). 8k/1k, InferenceX-aligned.
# Usage (inside container): bash pd_sweep.sh "2 32" 8100 <outdir> pd_intranode
set -u
CONCS="${1:-2 32}"; PORT="${2:-8100}"
OUTDIR="${3:-/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg/results}"
TAG="${4:-pd_intranode}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
ISL="${ISL:-8192}"; OSL="${OSL:-1024}"
mkdir -p "$OUTDIR"
for C in $CONCS; do
  NP=$((C * 10)); WU=$((C * 2))
  OUT="$OUTDIR/${TAG}_c${C}.jsonl"
  echo "=== [$TAG] conc=$C num_prompts=$NP warmup=$WU -> $OUT ==="
  python3 -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$C" --num-prompts "$NP" \
    --warmup-requests "$WU" --request-rate inf \
    --output-file "$OUT" 2>&1 | tail -25
  echo "=== conc=$C done ==="
done
