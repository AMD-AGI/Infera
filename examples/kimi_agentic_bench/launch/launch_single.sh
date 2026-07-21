#!/bin/bash
# Launch the single-node BASELINE engine (TP8, prefill+decode share one batch).
# Run inside the Infera container:  TP=8 bash launch/launch_single.sh
#
# No etcd discovery: the baseline serves the benchmark directly on its own port, so it
# does not register into the PD router's pool. Env toggles: TP (default 8), PORT, HOSTIP.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; source "$HERE/env.sh"
TP="${TP:-8}"; STATS="${STATS:-1}"; HOSTIP="${HOSTIP:-$PREFILL_IP}"; PORT="${PORT:-30000}"

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
       HSA_NO_SCRATCH_RECLAIM=1 PYTHONHASHSEED=0
export VLLM_HOST_IP=$HOSTIP

STATFLAG="--disable-log-stats"; [ "$STATS" = "1" ] && STATFLAG=""

cd /tmp
exec python3 -m infera.engine.vllm \
  --model "$MODEL" --served-model-name "$SERVED" \
  --port $PORT --host 0.0.0.0 --advertise-host $HOSTIP \
  --tensor-parallel-size $TP --enable-expert-parallel \
  --max-model-len 131072 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --block-size 16 \
  --enable-prefix-caching --trust-remote-code $STATFLAG
