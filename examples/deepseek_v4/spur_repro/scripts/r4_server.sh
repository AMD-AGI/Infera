#!/usr/bin/env bash
# R4 单机 SGLang DSv4 启动器 —— 复刻 legacy sglang_single_r4 的 R4 recipe。
# 两个关键杠杆: --attention-backend dsv4 + SGLANG_OPT_USE_FUSED_COMPRESS 全套 env。
# 在容器内运行: bash r4_server.sh [nodp|dp] [PORT] [CONC]
set -x
MODE="${1:-nodp}"
PORT="${2:-30000}"
CONC="${3:-32}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
ISL="${ISL:-8192}"
TP="${TP:-8}"

# ---- R4 env (legacy manifest 全套,一个都不能少) ----
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false

COMMON_ARGS=(
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code
  --host 127.0.0.1 --port "$PORT" --attention-backend dsv4
  --cuda-graph-max-bs 128 --disable-radix-cache --page-size 256
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion
  --mem-fraction-static 0.90
)

if [ "$MODE" = "dp" ]; then
  export SGLANG_DP_USE_GATHERV=1
  CHUNK=$((ISL * TP))
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --dp "$TP" --enable-dp-attention --ep-size "$TP" \
    --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 \
    --chunked-prefill-size "$CHUNK" \
    --max-running-requests "$CONC"
else
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --chunked-prefill-size "$ISL"
fi
