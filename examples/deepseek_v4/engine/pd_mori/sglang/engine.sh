#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE sglang PD leg (prefill|decode) via infera.engine.sglang over MoRI-IO RDMA.
# why : same DSv4 recipe as mooncake, but MoRI pairs NIC<->NIC by GID subnet so it needs ALL
#       active ionic NICs + MORI_IB_GID_INDEX=1. PD DPA rule: c>=128 on. callee of up.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
ROLE="${ROLE:?ROLE=prefill|decode}"; MY_IP="${MY_IP:?MY_IP=this node data-plane IP}"
ETCD_IP="${ETCD_IP:?ETCD_IP=etcd host (prefill node)}"; CTR="${CTR:-dsv4_pd_sgl_mori}"
TP="${TP:-8}"; BASE_GPU="${BASE_GPU:-0}"; PORT="${PORT:-30000}"; CONC="${CONC:-640}"
CHUNK="${CHUNK:-163840}"; BOOTSTRAP="${BOOTSTRAP:-8998}"; LOG="${LOG:-/tmp/pd_mori_${ROLE}_${PORT}.log}"
# Auto-detect the live GPU arch from amd-smi structured output (no awk/grep/sed);
# runs in the container where the ROCm stack lives. Override via GPU_ARCH.
GPU_ARCH="${GPU_ARCH:-$(docker exec "$CTR" amd-smi static -g 0 --asic --json \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["asic"]["target_graphics_version"])')}"

# MoRI wants every ACTIVE ionic NIC (opposite of mooncake, where you drop the flag).
IBDEVS=$(docker exec "$CTR" bash -lc 'for d in /sys/class/infiniband/*; do n=$(basename "$d"); \
  s=$(cat "$d/ports/1/state" 2>/dev/null); [[ "$s" == *ACTIVE* ]] && echo "$n"; done | sort -V | paste -sd,')
[ -n "$IBDEVS" ] || die "no active ionic NICs in $CTR — inject libionic first"

FUSED="SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0 SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false"
# MoRI RDMA env: GID 1 (ULA, not link-local); bind data-plane IP; generous cross-node timeouts.
# HSA_NO_SCRATCH_RECLAIM=1 frees GPU mem for RDMA MR reg; NIC pins gloo/mori to the data-plane rail.
NIC="${RDMA_NIC:-$(docker exec "$CTR" bash -lc "ip -o -4 addr show | awk '\$4 ~ /^$MY_IP\// {print \$2; exit}'" 2>/dev/null)}"
RDMA_ENV="MORI_IB_GID_INDEX=$GID_INDEX MORI_GPU_ARCHS=$GPU_ARCH NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_HOST_IP=$MY_IP HOST_IP=$MY_IP SGLANG_LOCAL_IP_NIC=$NIC GLOO_SOCKET_IFNAME=$NIC MORI_SOCKET_IFNAME=$NIC SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800"

if [ "$ROLE" = "prefill" ]; then MEMFRAC="0.85"; ROLE_FLAGS="--disaggregation-bootstrap-port $BOOTSTRAP"
else MEMFRAC="0.90"; ROLE_FLAGS="--no-enable-kv-events"; fi

COMMON="--model-path $INFERA_MODEL --tp-size $TP --trust-remote-code --host $MY_IP --port $PORT \
  --advertise-host $MY_IP --etcd-endpoint $ETCD_IP:$ETCD_PORT --discovery-backend etcd \
  --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache \
  --page-size 256 --cuda-graph-max-bs 512 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static $MEMFRAC --context-length 9472 --max-running-requests $CONC --base-gpu-id $BASE_GPU \
  --disaggregation-mode $ROLE --disaggregation-transfer-backend mori --disaggregation-ib-device $IBDEVS \
  --watchdog-timeout 3600"

if [ "$CONC" -ge 128 ]; then
  DPA_ENV="SGLANG_DP_USE_GATHERV=1"
  PARALLEL="--dp $TP --enable-dp-attention --ep-size $TP --enable-prefill-delayer \
    --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size $CHUNK --max-prefill-tokens $CHUNK"
else DPA_ENV=""; PARALLEL="--chunked-prefill-size 8192"; fi

docker exec -d "$CTR" bash -lc "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $FUSED $RDMA_ENV $DPA_ENV; \
  python3 -m infera.engine.sglang $COMMON $ROLE_FLAGS $PARALLEL > $LOG 2>&1"
log "sglang PD-mori $ROLE (TP$TP base$BASE_GPU :$PORT conc$CONC nics=$IBDEVS) launching -> $LOG"
