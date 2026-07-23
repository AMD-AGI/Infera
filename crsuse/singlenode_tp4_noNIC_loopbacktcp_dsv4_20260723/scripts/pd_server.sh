#!/bin/bash
# ============================================================================
# DSv4-Pro intra-node 1P1D PD server (prefill OR decode) on ONE spur 8-GPU node.
# Combines: legacy pd_1p1d PD wiring  +  my proven R4 single-node recipe (spur).
#
# MVP posture: NO RDMA / NO ionic  -> transfer backend = mooncake_tcp (loopback).
# prefill = TP4 GPU0-3 (:30000, bootstrap 8998); decode = TP4 GPU4-7 (:30100).
#
# Usage (inside container):
#   ROLE=prefill BASE_GPU=0 PORT=30000 bash pd_server.sh
#   ROLE=decode  BASE_GPU=4 PORT=30100 bash pd_server.sh
# ============================================================================
set -x
ROLE="${ROLE:?ROLE=prefill|decode}"
BACKEND="${BACKEND:-mooncake_tcp}"      # intra-node loopback: no RDMA needed
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
TP="${TP:-4}"
BASE_GPU="${BASE_GPU:-0}"
PORT="${PORT:-30000}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
P_IP="${P_IP:-127.0.0.1}"
D_IP="${D_IP:-127.0.0.1}"
ISL="${ISL:-8192}"
CTX="${CTX:-9472}"                      # 8k in + 1k out + headroom
CONC="${CONC:-32}"
MEMFRAC_P="${MEMFRAC_P:-0.85}"
MEMFRAC_D="${MEMFRAC_D:-0.90}"
WORK="${WORK:-/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg}"
LOG="${LOG:-$WORK/${ROLE}.log}"
mkdir -p "$(dirname "$LOG")"

if [ "$ROLE" = "prefill" ]; then MY_IP="$P_IP"; MEMFRAC="$MEMFRAC_P"; else MY_IP="$D_IP"; MEMFRAC="$MEMFRAC_D"; fi

# ---- R4 perf env (legacy manifest full set, verbatim; one cannot be missing) ----
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
# generous bootstrap/waiting timeouts (cold NFS weights can be slow)
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

COMMON=(
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4
  --cuda-graph-max-bs 128 --disable-radix-cache --page-size 256
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --max-running-requests "$CONC"
  --base-gpu-id "$BASE_GPU"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend "$BACKEND"
  --watchdog-timeout 3600
  --chunked-prefill-size "$ISL"
)
if [ "$ROLE" = "prefill" ]; then
  COMMON+=(--disaggregation-bootstrap-port "$BOOTSTRAP")
fi

python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1
