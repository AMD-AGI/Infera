#!/usr/bin/env bash
# CONTROL ARM: single-node colocated mix, NO DP-attention, NO disaggregation.
#
# Everything else held at the PD run's values (same patched image, same GLM DSA env, same
# ctx / per-rank chunk / cuda-graph-max-bs / TP8). The removed variables are exactly two:
#   1. PD disaggregation (no mooncake, no KV transfer, no separate decode leg)
#   2. DP-attention
# If conc=128 still produces the digit loop here -> not a PD bug, it is GLM-5.2 decode at
# high batch. If it is clean here -> the trigger needs PD and/or DPA.
#
# Runs inside the existing pd_uni container on the prefill node (kills its PD leg first).
set -euo pipefail
HOST="${HOST:-chi2867}"; IP="${IP:-10.2.122.44}"
CTR=pd_uni
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PORT="${PORT:-30000}"
TP=8
CTX="${CTX:-32768}"
CHUNK="${CHUNK:-8192}"          # == the PD run's PER-RANK chunk (65536/dp8) -> same compute shape
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
GMU="${GMU:-0.88}"
KIT=/mnt/vast/c_huggingface/glm52_longctx_pd
LOG="$KIT/single_nodpa_${PORT}.log"
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 \"$2\""; }

echo "== 0. free the node (kill PD leg + router; no nested single quotes -- see REPRODUCE pitfalls) =="
J "$HOST" "docker exec $CTR pkill -9 -f launch_server; docker exec $CTR pkill -9 -f sglang_router; true" || true
# host-side router: its comm is `sglang::router`, so `pkill -f sglang_router` misses it.
J "$HOST" "pkill -9 -x sglang::router; true" || true
sleep 25

echo "== 1. launch single-node colocated, DPA=0, no disaggregation =="
J "$HOST" "docker exec -d $CTR env \
  SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0 \
  SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash -c 'python3 -m sglang.launch_server --model-path $MODEL --served-model-name $SERVED \
    --host 0.0.0.0 --port $PORT --tp-size $TP --trust-remote-code \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static $GMU \
    --context-length $CTX --chunked-prefill-size $CHUNK \
    --max-running-requests $MAX_RUNNING --cuda-graph-max-bs $CUDA_GRAPH_BS \
    --watchdog-timeout 3600 > $LOG 2>&1'"

echo "== launched. no-DPA cold start ~4-8 min. watch: $LOG -> 'ready to roll'"
echo "   then hit it DIRECTLY at http://$IP:$PORT  (no router: this is not a PD leg)"
