#!/usr/bin/env bash
# GLM-5.2-MXFP4 single-node on lmsysorg/sglang:v0.5.16-rocm720-mi35x.
# ONE variable: DSA_BACKEND (aiter under test | tilelang known-good baseline).
# Everything else = proven exp01 recipe. Small ctx = MVP, correctness/smoke only.
set -uo pipefail
IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.16-rocm720-mi35x}
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PORT="${PORT:-30000}"
DSA_BACKEND="${DSA_BACKEND:-aiter}"
TP="${TP:-8}"
CTX="${CTX:-32768}"
MAXRUN="${MAXRUN:-64}"
CGBS="${CGBS:-64}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-${DSA_BACKEND}}"
NAME="${NAME:-glm52-v0516-${TAG}}"
OUT=/mnt/vast/c_huggingface/glm52_dsa_v0516
mkdir -p "$OUT"

docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 3

docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=1 -e SGLANG_OPT_USE_TOPK_V2=0 \
  -e SGLANG_OPT_USE_JIT_NORM=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint python3 "$IMAGE" \
  -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP" --trust-remote-code \
    --dsa-prefill-backend "$DSA_BACKEND" --dsa-decode-backend "$DSA_BACKEND" \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
    --context-length "$CTX" --chunked-prefill-size 8192 \
    --max-running-requests "$MAXRUN" --cuda-graph-max-bs "$CGBS" \
    $EXTRA >/dev/null 2>&1

echo "started $NAME  backend=$DSA_BACKEND tp=$TP ctx=$CTX maxrun=$MAXRUN cgbs=$CGBS extra='$EXTRA'"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
echo "log: docker logs -f $NAME   |   archive -> $OUT/${TAG}.log"
