#!/usr/bin/env bash
# Single-node GLM-5.2-MXFP4 with DP-attention (DPA) ON, long context (>=128K) so a 65K-token
# input fits. Recipe = packup exp01 GLM DSA base + exp07 DPA block (dp8/ep8/GATHERV), minus the
# PD/mooncake parts (this is a colocated single node).
set -uo pipefail
IMAGE="${IMAGE:-rocm/infera:sglang-v0.1.0-rc6}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
NAME="${NAME:-glm52-dpa-longctx}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DPA="${DPA:-1}"
CTX="${CTX:-131072}"                 # must exceed the 65K probe input + generation
CHUNK="${CHUNK:-16384}"              # long-prefill chunk; DPA shards batch across ranks
GMU="${GMU:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-64}"
LOG_HOST="${LOG_HOST:-/mnt/vast/c_huggingface/glm52_longctx}"
mkdir -p "$LOG_HOST"

DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
fi

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 \
  -e SGLANG_DP_USE_GATHERV="$DPA" \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint python3 "$IMAGE" \
  -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP" --trust-remote-code \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" \
    --context-length "$CTX" --chunked-prefill-size "$CHUNK" \
    --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
    --watchdog-timeout 3600 \
    "${DP_ARGS[@]}"
echo "started $NAME dpa=$DPA ctx=$CTX chunk=$CHUNK gmu=$GMU"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
