#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch one atom mix worker via infera.engine.atom (aligned to the tuned recipe).
# why : fp8 KV + cudagraph-capture-sizes + official env are perf-critical; conc picks
#       plain/DPA/DPA+TBO. NEVER set ATOM_USE_TRITON_MOE (crashes this image). callee of run.sh.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
CTR="${CTR:-dsv4_atom_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"; CONC="${CONC:-64}"; ISL="${ISL:-8192}"
LOG="/tmp/atom_mix.log"
CGSIZES='[1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512]'

# conc-based variant (8k1k: <=64 plain / 128 DPA / >=256 DPA+TBO). TBO adds GPU_MAX_HW_QUEUES=5.
PARALLEL="-tp 8"; TBO_ENV=""
if [ "$CONC" -ge 256 ]; then PARALLEL="-tp 8 --enable-dp-attention --enable-tbo"; TBO_ENV="GPU_MAX_HW_QUEUES=5"
elif [ "$CONC" -ge 128 ]; then PARALLEL="-tp 8 --enable-dp-attention"; fi

docker exec -d "$CTR" bash -lc "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 HF_HOME=/tmp/hf \
  ATOM_DISABLE_MMAP=true AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING $TBO_ENV; \
  python3 -m infera.engine.atom --model $INFERA_MODEL --server-port $ENGINE_PORT --host 0.0.0.0 \
    --advertise-host $NODE_IP --etcd-endpoint $NODE_IP:$ETCD_PORT \
    $PARALLEL --kv_cache_dtype fp8 --trust-remote-code \
    --gpu-memory-utilization 0.9 --no-enable_prefix_caching \
    --cudagraph-capture-sizes '$CGSIZES' --max-model-len 16384 > $LOG 2>&1"
log "atom mix worker launching (conc=$CONC) -> $LOG"
wait_worker_log "$CTR" "$LOG" 200 || die "atom mix worker failed to become ready"
