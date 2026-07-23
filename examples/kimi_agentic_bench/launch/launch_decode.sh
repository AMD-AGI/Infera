#!/bin/bash
# Launch the DECODE engine (Infera PD, kv_consumer role) on the decode node.
# Run inside the Infera container:  bash launch/launch_decode.sh
#
# Decode-only => a pure-decode full CUDA graph => low, stable inter-token latency.
# Default is TP2 x DP4 (8 GPUs). Env toggles: TP (default 2), DP (default 4).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; source "$HERE/env.sh"
TP="${TP:-2}"; DP="${DP:-4}"; PORT="${PORT:-30000}"

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
       HSA_NO_SCRATCH_RECLAIM=1 PYTHONHASHSEED=0
export VLLM_HOST_IP=$DECODE_IP MC_GID_INDEX=1 NCCL_IB_GID_INDEX=1 RDMAV_FORK_SAFE=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=${MC_PORT:-8998}

DPFLAG="--data-parallel-size $DP --data-parallel-size-local $DP"; [ "$DP" = "1" ] && DPFLAG=""

cd /tmp
exec python3 -m infera.engine.vllm \
  --model "$MODEL" --served-model-name "$SERVED" \
  --port $PORT --host 0.0.0.0 --advertise-host $DECODE_IP --etcd-endpoint $ETCD_EP \
  $DPFLAG --tensor-parallel-size $TP --enable-expert-parallel \
  --max-model-len 131072 --max-num-seqs 256 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization ${GMU:-0.85} --kv-cache-dtype fp8 --block-size 16 \
  --enable-prefix-caching --trust-remote-code \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
