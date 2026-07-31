#!/usr/bin/env bash
# CONTROL ARM for the PD first-touch corruption bug.
#
# Single-node colocated, but otherwise IDENTICAL to the PD DPA=0 leg: same pd-unified image,
# same ctx/chunk/gmu/TP, same GLM DSA env. The ONLY removed variable is disaggregation
# (no --disaggregation-mode, no mooncake, no KV transfer).
#
# If novel unseen shapes corrupt here too -> the bug is in the tilelang DSA path, not PD.
# If they are clean here -> the bug is PD-specific.
set -uo pipefail
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
NAME="${NAME:-glm52-single-uni}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DPA="${DPA:-0}"                      # match the PD control arm (DPA=0)
CTX="${CTX:-131072}"
CHUNK="${CHUNK:-16384}"              # same as the PD DPA=0 leg resolved to
GMU="${GMU:-0.88}"                   # same as the PD prefill leg
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-64}"
KIT=/mnt/vast/c_huggingface/glm52_longctx_pd
LOG="$KIT/single_uni_${PORT}.log"

DP_ARGS=()
[ "$DPA" = "1" ] && DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 \
  -e SGLANG_DP_USE_GATHERV="$DPA" \
  -e SAFETENSORS_FAST_GPU=1 -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint bash "$IMAGE" -c "
    python3 -m sglang.launch_server \
      --model-path '$MODEL' --served-model-name '$SERVED' \
      --host 0.0.0.0 --port $PORT --tp-size $TP --trust-remote-code \
      --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
      --kv-cache-dtype fp8_e4m3 --mem-fraction-static $GMU \
      --context-length $CTX --chunked-prefill-size $CHUNK \
      --max-running-requests $MAX_RUNNING --cuda-graph-max-bs $CUDA_GRAPH_BS \
      --watchdog-timeout 3600 ${DP_ARGS[*]} > '$LOG' 2>&1"
echo "started $NAME dpa=$DPA ctx=$CTX chunk=$CHUNK gmu=$GMU -> $LOG"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
