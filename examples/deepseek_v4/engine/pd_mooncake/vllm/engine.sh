#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE vllm PD leg (prefill|decode) via infera.engine.vllm over mooncake (converted
# from native `vllm serve` + vllm-router proxy). why: infera.server auto-pairs via etcd, no static
# proxy. KV via --kv-transfer-config MooncakeConnector (producer/consumer). callee of up.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
ROLE="${ROLE:?ROLE=prefill|decode}"; MY_IP="${MY_IP:?MY_IP=this node data-plane IP}"
ETCD_IP="${ETCD_IP:?ETCD_IP=etcd host (prefill node)}"; CTR="${CTR:-dsv4_pd_vllm}"
LOG="${LOG:-/tmp/pd_vllm_${ROLE}.log}"

# role -> kv_role + distinct mooncake bootstrap port (producer 8998 / consumer 8999).
if [ "$ROLE" = "prefill" ]; then KV_ROLE=kv_producer; BOOT=8998; else KV_ROLE=kv_consumer; BOOT=8999; fi
KVCFG="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"$KV_ROLE\"}"

# aiter env + PD-specifics (VLLM_HOST_IP = data-plane IP; MHC_BACKEND=aiter for CG capture; GID 1).
ENVSET="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 AITER_BF16_FP8_MOE_BOUND=0 \
PYTHONHASHSEED=0 VLLM_MHC_BACKEND=aiter MC_GID_INDEX=$GID_INDEX VLLM_HOST_IP=$MY_IP VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOT"

docker exec -d "$CTR" bash -lc "export $ENVSET; python3 -m infera.engine.vllm --model $INFERA_MODEL \
  --tensor-parallel-size 8 --distributed-executor-backend mp --kv-cache-dtype fp8 --moe-backend aiter \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.6 --max-num-batched-tokens 8192 --max-model-len 9472 --max-num-seqs 32 \
  --compilation-config {\\\"max_cudagraph_capture_size\\\":32} --disable-hybrid-kv-cache-manager \
  --trust-remote-code --host 0.0.0.0 --port $ENGINE_PORT --advertise-host $MY_IP \
  --etcd-endpoint $ETCD_IP:$ETCD_PORT --discovery-backend etcd --request-transport http \
  --kv-event-transport zmq --kv-transfer-config '$KVCFG' > $LOG 2>&1"
log "vllm PD $ROLE (kv=$KV_ROLE boot=$BOOT) launching -> $LOG"
