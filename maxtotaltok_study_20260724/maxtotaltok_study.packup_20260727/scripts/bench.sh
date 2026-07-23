#!/bin/bash
# Send 8k/1k conc=128 load through the router. Run INSIDE mtt_pd on prefill node.
set -u
P_IP="${P_IP:?prefill IP (router host)}"
RPORT="${RPORT:-8100}"
MODEL="${MODEL:-/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro}"
CONC="${CONC:-128}"
NPROMPT="${NPROMPT:-$((CONC*10))}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
TAG="${TAG:-c${CONC}}"
OUT="${OUT:-/mnt/vast/c_huggingface/mtt_study}"
LOG="${LOG:-$OUT/bench_${TAG}_$(date +%H%M%S).log}"
mkdir -p "$OUT"
echo "[bench] tag=$TAG conc=$CONC nprompt=$NPROMPT isl=$ISL osl=$OSL -> $LOG"
# NOTE: this 0.5.13 image installs sglang as a PEP-420 namespace pkg where the
# benchmark/ subtree is not on the import path -> `-m sglang.bench_serving` fails
# with ModuleNotFoundError sglang.benchmark.datasets. Point PYTHONPATH at the
# editable source python/ dir so the whole tree resolves.
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
python3 -m sglang.bench_serving --backend sglang-oai --base-url "http://$P_IP:$RPORT" \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
  --max-concurrency "$CONC" --num-prompts "$NPROMPT" --warmup-requests $((CONC/4)) \
  --request-rate inf --output-file "$OUT/bench_${TAG}.jsonl" > "$LOG" 2>&1
echo "[bench] done. tail:"; tail -25 "$LOG"
