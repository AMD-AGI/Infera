#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch one sglang mix worker (both phases) via infera.engine.sglang.
# why : two levers (--attention-backend dsv4 + fused-compress env) or throughput halves;
#       variant by conc — nodp pure TP8 (c<=128) / dp DP-attn (c>=256). callee of run.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
CTR="${CTR:-dsv4_sgl_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"; CONC="${CONC:-64}"
LOG="/tmp/sgl_mix.log"

# Two-lever env set — all entries are required for full throughput (note: do not drop any).
FUSED="SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0 SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false"

COMMON="--model-path $INFERA_MODEL --tp-size 8 --trust-remote-code --host 0.0.0.0 --port $ENGINE_PORT \
  --advertise-host $NODE_IP --etcd-endpoint $NODE_IP:$ETCD_PORT --discovery-backend etcd \
  --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache \
  --page-size 256 --cuda-graph-max-bs 128 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static 0.90"

# conc>=256 -> DP-attention variant (adds GATHERV + dp/ep + prefill-delayer + big chunk).
if [ "$CONC" -ge 256 ]; then
  VARIANT="SGLANG_DP_USE_GATHERV=1"; PARALLEL="--dp 8 --enable-dp-attention --ep-size 8 \
    --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size 65536 --max-running-requests 256"
else
  VARIANT=""; PARALLEL="--chunked-prefill-size 8192"
fi

docker exec -d "$CTR" bash -lc "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $FUSED $VARIANT; \
  python3 -m infera.engine.sglang $COMMON $PARALLEL > $LOG 2>&1"
log "sglang mix worker launching (conc=$CONC, $([ $CONC -ge 256 ] && echo dp || echo nodp)) -> $LOG"
wait_worker_log "$CTR" "$LOG" 200 || die "sglang mix worker failed to become ready"
