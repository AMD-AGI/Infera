#!/usr/bin/env bash
# Single-node GLM-5.2-MXFP4: test whether the DSA MAIN attention kernel works with aiter
# (vs our proven tilelang). ONE variable changed: --dsa-prefill/decode-backend. Everything
# else = proven exp01 recipe (indexer env, fp8 kv, TP8). Runs on ONE node inside pd-unified.
set -uo pipefail
IMAGE=infera/engine-sglang:pd-unified
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PORT="${PORT:-30000}"
NAME="${NAME:-glm52-dsa-test}"
TP="${TP:-8}"
DSA_BACKEND="${DSA_BACKEND:-aiter}"   # aiter (under test) | tilelang (proven baseline)
CTX="${CTX:-32768}"                    # small ctx MVP; correctness+smoke only
LOG="${LOG:-/mnt/vast/c_huggingface/glm52_dsa_test/${DSA_BACKEND}.log}"
mkdir -p "$(dirname "$LOG")"

docker rm -f "$NAME" >/dev/null 2>&1 || true
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
    --max-running-requests 64 --cuda-graph-max-bs 64 \
  >/dev/null 2>&1
echo "started $NAME (dsa-backend=$DSA_BACKEND, log inside container stdout; server log -> docker logs)"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
