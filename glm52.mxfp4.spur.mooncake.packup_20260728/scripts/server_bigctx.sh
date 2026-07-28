#!/bin/bash
# Single-node GLM-5.2-MXFP4 server, large context for AgenticBench caseA (up to 260K input).
# Plain sglang.launch_server (no PD), TP8, GLM DSA-ROCm recipe, cache-report on.
set -u
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
MY_IP="${MY_IP:?MY_IP}"
PORT="${PORT:-30000}"
CTX="${CTX:-262144}"
LOG="${LOG:-/home/yihou/glm52_spur/logs/server_bigctx.log}"

export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size 8 --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.90 --context-length "$CTX" \
  --chunked-prefill-size 16384 --enable-cache-report \
  --max-running-requests 48 --cuda-graph-max-bs 48 --watchdog-timeout 3600 \
  > "$LOG" 2>&1
