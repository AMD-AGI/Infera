#!/usr/bin/env bash
# GLM-5.2-FP8 (block-fp8 141 shards) on v0.5.16-rocm720-mi35x.
# Single variable vs the MXFP4 runs: the model. DSA/attention flags identical.
set -uo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.16-rocm720-mi35x}
MODEL=${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-FP8}
PORT="${PORT:-30000}"
DSA_BACKEND="${DSA_BACKEND:-aiter}"
TP="${TP:-8}"
CTX="${CTX:-32768}"
MAXRUN="${MAXRUN:-64}"
CGBS="${CGBS:-64}"
MEMFRAC="${MEMFRAC:-0.85}"
TRACE="${TRACE:-1}"          # in-place __DSA_TRACE__ patch
EXTRA="${EXTRA:-}"
TAG="${TAG:-fp8_${DSA_BACKEND}}"
NAME="${NAME:-glm52-${TAG}}"
mkdir -p /mnt/vast/c_huggingface/glm52_dsa_v0516

docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 3

docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast -v /tmp/dsa_patch_trace.py:/tmp/dsa_patch_trace.py \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e SAFETENSORS_FAST_GPU=1 -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint bash "$IMAGE" -lc "
    if [ '$TRACE' = '1' ]; then python3 /tmp/dsa_patch_trace.py || echo 'TRACE PATCH FAILED (continuing)'; fi
    exec python3 -m sglang.launch_server \
      --model-path $MODEL --served-model-name glm5.2-fp8 \
      --host 0.0.0.0 --port $PORT --tp-size $TP --trust-remote-code \
      --dsa-prefill-backend $DSA_BACKEND --dsa-decode-backend $DSA_BACKEND \
      --kv-cache-dtype fp8_e4m3 --mem-fraction-static $MEMFRAC \
      --context-length $CTX --chunked-prefill-size 8192 \
      --max-running-requests $MAXRUN --cuda-graph-max-bs $CGBS $EXTRA
  " >/dev/null 2>&1

echo "started $NAME  model=$(basename $MODEL) backend=$DSA_BACKEND tp=$TP ctx=$CTX memfrac=$MEMFRAC trace=$TRACE"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
