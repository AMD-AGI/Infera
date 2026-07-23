#!/bin/bash
# bench_serving sweep through the PD router (mlx5 KV path). Run INSIDE the prefill
# container. Context controlled: ISL+OSL < server context_length (9472).
#   ISL=4096 OSL=512 -> 4608 < 9472 (safe headroom for all conc points).
# Usage: bash bench_sweep_mlx5.sh "1 16 32 64 128 256" 8200 <outdir>
set -u
CONCS="${1:-1 16 32 64 128 256}"
PORT="${2:-8200}"
OUTDIR="${3:?outdir}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
ISL="${ISL:-4096}"; OSL="${OSL:-512}"
NP_CAP="${NP_CAP:-512}"     # cap num-prompts (8k-ish inputs tokenize slowly client-side)
mkdir -p "$OUTDIR"
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

for C in $CONCS; do
  NP=$((C * 5)); [ "$NP" -lt 8 ] && NP=8; [ "$NP" -gt "$NP_CAP" ] && NP=$NP_CAP
  WU=$((C < 4 ? 2 : C / 2))
  OUT="$OUTDIR/mlx5_c${C}.jsonl"
  echo "=== conc=$C num_prompts=$NP warmup=$WU ISL=$ISL OSL=$OSL -> $OUT ==="
  python3 -m sglang.benchmark.serving --backend sglang \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$C" --num-prompts "$NP" \
    --warmup-requests "$WU" --request-rate inf \
    --output-file "$OUT" 2>&1 | tail -35
  echo "=== conc=$C done ==="
done
echo "SWEEP_DONE"
