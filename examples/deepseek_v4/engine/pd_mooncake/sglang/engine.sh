#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE sglang PD leg (prefill|decode) via infera.engine.sglang over mooncake RDMA.
# why : tuned recipe (two levers + DP-attn + chunk 163840 + role-asymmetric mem-frac).
#       PD DPA rule: c>=128 on. callee of up.sh (per node).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
ROLE="${ROLE:?ROLE=prefill|decode}"; MY_IP="${MY_IP:?MY_IP=this node data-plane IP}"
ETCD_IP="${ETCD_IP:?ETCD_IP=etcd host (prefill node)}"; CTR="${CTR:-dsv4_pd_sgl}"
BACKEND="${BACKEND:-mooncake}"; TP="${TP:-8}"; BASE_GPU="${BASE_GPU:-0}"; PORT="${PORT:-30000}"
CONC="${CONC:-640}"; CHUNK="${CHUNK:-163840}"; BOOTSTRAP="${BOOTSTRAP:-8998}"
LOG="${LOG:-/tmp/pd_sgl_${ROLE}_${PORT}.log}"
# data-plane NIC: pins RDMA/gloo to the rail (auto-detect the iface holding MY_IP if unset).
NIC="${RDMA_NIC:-$(docker exec "$CTR" bash -lc "ip -o -4 addr show | awk '\$4 ~ /^$MY_IP\// {print \$2; exit}'" 2>/dev/null)}"

FUSED="SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0 SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false"
# cross-node bootstrap over cold NFS weights needs long timeouts; MC_GID_INDEX=1 = RoCEv2 (not link-local).
# note: HSA_NO_SCRATCH_RECLAIM=1 frees GPU mem for mooncake RDMA MR registration; NIC pins the rail.
RDMA_ENV="MC_GID_INDEX=$GID_INDEX SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800 SGLANG_HOST_IP=$MY_IP HOST_IP=$MY_IP SGLANG_LOCAL_IP_NIC=$NIC GLOO_SOCKET_IFNAME=$NIC NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1"

# prefill mem-frac 0.85 (DP-attn prefill needs the headroom); decode 0.90. decode drops kv-events (SWA).
if [ "$ROLE" = "prefill" ]; then MEMFRAC="0.85"; ROLE_FLAGS="--disaggregation-bootstrap-port $BOOTSTRAP"
else MEMFRAC="0.90"; ROLE_FLAGS="--no-enable-kv-events"; fi

COMMON="--model-path $INFERA_MODEL --tp-size $TP --trust-remote-code --host $MY_IP --port $PORT \
  --advertise-host $MY_IP --etcd-endpoint $ETCD_IP:$ETCD_PORT --discovery-backend etcd \
  --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache \
  --page-size 256 --cuda-graph-max-bs 512 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static $MEMFRAC --context-length 9472 --max-running-requests $CONC --base-gpu-id $BASE_GPU \
  --disaggregation-mode $ROLE --disaggregation-transfer-backend $BACKEND --watchdog-timeout 3600"

# PD DPA rule: c>=128 enable DP-attn (dp=ep=TP); else pure TP. GATHERV needed for DP-attn.
if [ "$CONC" -ge 128 ]; then
  DPA_ENV="SGLANG_DP_USE_GATHERV=1"
  PARALLEL="--dp $TP --enable-dp-attention --ep-size $TP --enable-prefill-delayer \
    --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size $CHUNK --max-prefill-tokens $CHUNK"
else DPA_ENV=""; PARALLEL="--chunked-prefill-size 8192"; fi

docker exec -d "$CTR" bash -lc "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $FUSED $RDMA_ENV $DPA_ENV; \
  python3 -m infera.engine.sglang $COMMON $ROLE_FLAGS $PARALLEL > $LOG 2>&1"
log "sglang PD $ROLE (TP$TP base$BASE_GPU :$PORT conc$CONC) launching -> $LOG"
