#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE atom PD leg (prefill|decode) via infera.engine.atom over mooncake RDMA.
# why : reuses the validated atom mix recipe (fp8 KV + cudagraph sizes + official env) + PD via
#       --kv-transfer-config (producer/consumer). NEVER set ATOM_USE_TRITON_MOE. callee of up.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
ROLE="${ROLE:?ROLE=prefill|decode}"; MY_IP="${MY_IP:?MY_IP=this node data-plane IP}"
ETCD_IP="${ETCD_IP:?ETCD_IP=etcd host (prefill node)}"; CTR="${CTR:-dsv4_pd_atom}"; TP="${TP:-8}"
LOG="${LOG:-/tmp/pd_atom_${ROLE}.log}"
CGSIZES='[1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512]'
IB=$(docker exec "$CTR" bash -lc 'for d in /sys/class/infiniband/*; do n=$(basename "$d"); \
  s=$(cat "$d/ports/1/state" 2>/dev/null); [[ "$s" == *ACTIVE* ]] && echo "$n"; done | sort -V | paste -sd,')
[ -n "$IB" ] || die "no active ionic NICs in $CTR — inject libionic first"

# role -> kv_role + distinct server/handshake ports (atom threads kv_transfer on completions path).
if [ "$ROLE" = "prefill" ]; then KV_ROLE=kv_producer; SPORT=30001; HPORT=6301
else KV_ROLE=kv_consumer; SPORT=30002; HPORT=6311; fi
KVCFG="{\"kv_role\":\"$KV_ROLE\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HPORT,\"http_port\":$SPORT,\"proxy_ip\":\"$MY_IP\",\"ib_device\":\"$IB\"}"

docker exec -d "$CTR" bash -lc "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 HF_HOME=/tmp/hf \
  ATOM_DISABLE_MMAP=true AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING \
  ATOM_HOST_IP=$MY_IP MC_DISABLE_HIP_TRANSPORT=1 RDMAV_FORK_SAFE=1 MC_GID_INDEX=$GID_INDEX; \
  python3 -m infera.engine.atom --model $INFERA_MODEL --server-port $SPORT --host 0.0.0.0 \
    --advertise-host $MY_IP --etcd-endpoint $ETCD_IP:$ETCD_PORT \
    -tp $TP --kv_cache_dtype fp8 --trust-remote-code --max-model-len 16384 \
    --max-num-seqs 256 --gpu-memory-utilization 0.90 --no-enable_prefix_caching \
    --cudagraph-capture-sizes '$CGSIZES' --kv-transfer-config '$KVCFG' > $LOG 2>&1"
log "atom PD $ROLE (kv=$KV_ROLE :$SPORT nics=$IB) launching -> $LOG"
