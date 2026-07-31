#!/usr/bin/env bash
# GLM-5.2 mooncake-PD-DPA concurrency sweep client. Runs INSIDE the prefill pd_uni container,
# hits the router (:8002). 1k/1k random. One jsonl per conc. Mirrors the DSv4 r4_sweep cadence
# (num_prompts = conc*N, warmup = conc*M) but capped so the biggest points still finish.
set -u
CONCS="${CONCS:-64 128 256 512 1024 2048}"
BASE="${BASE:-http://10.2.122.3:8002}"
MODEL="${MODEL:-glm5.2-mxfp4}"
TOK="${TOK:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
OUTDIR="${OUTDIR:-/mnt/vast/c_huggingface/glm52_p2b/sweep_dpa}"
TAG="${TAG:-dpa}"
mkdir -p "$OUTDIR"
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

for C in $CONCS; do
  # prompts = 4x conc (enough steady-state), warmup = min(conc, 64). Cap prompts at 4096 for the
  # huge points so a single sweep stays bounded (~tens of min total).
  NP=$((C * 4)); [ "$NP" -gt 4096 ] && NP=4096
  WU=$C; [ "$WU" -gt 64 ] && WU=64
  OUT="$OUTDIR/${TAG}_c${C}.jsonl"
  LOG="$OUTDIR/${TAG}_c${C}.log"
  echo "=== [$TAG] conc=$C prompts=$NP warmup=$WU isl=$ISL osl=$OSL -> $OUT ==="
  python3 -m sglang.bench_serving --backend sglang-oai \
    --base-url "$BASE" --model "$MODEL" --tokenizer "$TOK" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$C" --num-prompts "$NP" \
    --warmup-requests "$WU" --request-rate inf \
    --output-file "$OUT" 2>&1 | tee "$LOG" | tail -34
  echo "=== conc=$C done ($(date +%H:%M:%S)) ==="
  sleep 5
done
echo "=== SWEEP COMPLETE -> $OUTDIR ==="
