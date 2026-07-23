#!/bin/bash
# Single-node TP4 P <-> TP4 D on ONE 8-GPU node, KV over Mooncake HIP transport
# (intra-node GPU<->GPU via hipIpcOpenMemHandle / XGMI — NOT the RDMA NIC).
# Run INSIDE the container. Two legs share the node: P=GPU0-3, D=GPU4-7.
#
# Key: MC_ENABLE_HIP_TRANSPORT=1 un-gates installTransport("hip"); selectTransport
# then prefers hip (prio 4) over rdma (prio 2). KV moves over XGMI, no NIC data path.
set -x
ROLE="${ROLE:?ROLE=prefill|decode}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
MY_IP="${MY_IP:?MY_IP=this node IP (http/bootstrap plane)}"
PORT="${PORT:?PORT}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
TP="${TP:-4}"
BASE_GPU="${BASE_GPU:?BASE_GPU=0 for P, 4 for D}"
CONC="${CONC:-64}"
MEMFRAC="${MEMFRAC:-0.42}"   # TP4 => weights ~159GB spread over 4 GPUs = ~40GB/GPU only?
CTX="${CTX:-9472}"
LOG="${LOG:-/tmp/tp4_${ROLE}_${PORT}.log}"

# --- HIP transport ON (the whole point) ---
export MC_ENABLE_HIP_TRANSPORT=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# R4 perf env
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false

COMMON=(
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4
  --base-gpu-id "$BASE_GPU"
  --cuda-graph-max-bs 256 --disable-radix-cache --page-size 256
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --max-running-requests "$CONC"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake
  --watchdog-timeout 3600 --chunked-prefill-size 8192
)
if [ "$ROLE" = "prefill" ]; then
  COMMON+=(--disaggregation-bootstrap-port "$BOOTSTRAP")
fi

python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1
