#!/usr/bin/env bash
# Launch aiter DSA and, instead of guessing, capture WHAT the code actually does at the
# failing bs: enable a python-level trace of the aiter metadata/decode calls plus
# AMD_SERIALIZE_KERNEL so the fault is attributed to the exact kernel launch.
set -uo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.16-rocm720-mi35x}
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
NAME="${NAME:-glm52-trace}"
PORT="${PORT:-30000}"
CGBS="${CGBS:-64}"
MAXRUN="${MAXRUN:-64}"
CTX="${CTX:-32768}"

docker rm -f "$NAME" >/dev/null 2>&1 || true; sleep 3
docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast -v /tmp/trace_hook.py:/tmp/trace_hook.py \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e SAFETENSORS_FAST_GPU=1 -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e AMD_SERIALIZE_KERNEL=3 \
  -e PYTHONFAULTHANDLER=1 \
  --entrypoint python3 "$IMAGE" /tmp/trace_hook.py \
    --model-path "$MODEL" --served-model-name glm5.2-mxfp4 \
    --host 0.0.0.0 --port "$PORT" --tp-size 8 --trust-remote-code \
    --dsa-prefill-backend aiter --dsa-decode-backend aiter \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --context-length "$CTX" --chunked-prefill-size 8192 \
    --max-running-requests "$MAXRUN" --cuda-graph-max-bs "$CGBS" \
  >/dev/null 2>&1
echo "started $NAME (AMD_SERIALIZE_KERNEL=3, aiter call trace)"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
