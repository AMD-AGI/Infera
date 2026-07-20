#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch one vllm mix worker via infera.engine.vllm (converted from native `vllm serve`).
# why : DP8+EP + full aiter env + fp8 KV are the tuned recipe; conc>=256 uses DP8, else TP8.
#       PYTHONHASHSEED=0 is mandatory. callee of run.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
CTR="${CTR:-dsv4_vllm_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"; CONC="${CONC:-256}"
TOK="${INFERA_TOKENIZER:-$INFERA_MODEL}"   # override when a checkpoint's own tokenizer is broken
LOG="/tmp/vllm_mix.log"

# aiter env set — tuned recipe. Note: PYTHONHASHSEED=0 is required for cross-restart cache hits.
AITER="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 AITER_BF16_FP8_MOE_BOUND=0 \
PYTHONHASHSEED=0 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 VLLM_ROCM_AITER_MOE_DISPATCH_POLICY=2 \
HIP_FORCE_DEV_KERNARG=1 VLLM_ROCM_USE_AITER_TRITON_ROPE=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1"

COMMON="--model $INFERA_MODEL --tokenizer $TOK --distributed-executor-backend mp --kv-cache-dtype fp8 --moe-backend aiter \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --no-enable-prefix-caching \
  --max-num-batched-tokens 8192 --max-model-len 9472 --max-num-seqs 512 \
  --compilation-config {\\\"max_cudagraph_capture_size\\\":128} --async-scheduling --trust-remote-code \
  --host 0.0.0.0 --port $ENGINE_PORT --advertise-host $NODE_IP --etcd-endpoint $NODE_IP:$ETCD_PORT \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq"

# conc>=256 -> DP8+EP (gmu 0.86 to survive c512); else TP8 (gmu 0.90).
if [ "$CONC" -ge 256 ]; then PARALLEL="--data-parallel-size 8 --enable-expert-parallel --gpu-memory-utilization 0.86"
else PARALLEL="--tensor-parallel-size 8 --gpu-memory-utilization 0.90"; fi

docker exec -d "$CTR" bash -lc "export $AITER; python3 -m infera.engine.vllm $COMMON $PARALLEL > $LOG 2>&1"
log "vllm mix worker launching (conc=$CONC, $([ $CONC -ge 256 ] && echo dp8 || echo tp8)) -> $LOG"
wait_worker_log "$CTR" "$LOG" 250 || die "vllm mix worker failed to become ready"
