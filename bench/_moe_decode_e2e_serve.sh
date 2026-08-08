#!/usr/bin/env bash
# Dedicated serve driver for the infera_decode MoE-experts e2e experiment (issue #40).
# Usage: _moe_decode_e2e_serve.sh <kernel|baseline> <port>
set -euo pipefail
MODE="${1:?kernel|baseline}"
PORT="${2:-8012}"
MODEL=/mnt/vast/john/huggingface/Qwen3.5-35B-A3B

export PYTHONPATH=/mnt/vast/jiejing/workspace/Optimus
export INFERA_MOE_DECODE_DEBUG=1
export INFERA_MOE_FIRE_FILE=/tmp/infera_moe_fires.txt
export VLLM_ROCM_USE_AITER=0        # plain (unshuffled) MoE weight layout
export INFERA_MOE_DECODE_MAX_TOKENS=16

if [[ "$MODE" == "kernel" ]]; then
  export INFERA_MOE_EXPERTS=infera_decode
  export INFERA_VLLM_OPS_DISABLE=0
else
  # baseline: builtin experts (aiter). Seam installs but selects no variant.
  export INFERA_MOE_EXPERTS=builtin
  export INFERA_VLLM_OPS_DISABLE=0
fi

cd /root
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen35 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --no-enable-log-requests \
  --port "$PORT"
