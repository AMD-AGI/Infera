#!/bin/bash
# Single-node TP4 P <-> TP4 D, KV over RDMA loopback (NOT hip transport).
# Parametric NIC: NIC_DEV=mlx5_0 (has ODP, dmabuf dynamic-attach, no pin) OR
#                 NIC_DEV=ionic_0 (no ODP, dmabuf pins -> expected double/crash).
# NO MC_ENABLE_HIP_TRANSPORT — KV goes over the RDMA NIC even though same-node.
# Run INSIDE the container.
set -x
ROLE="${ROLE:?ROLE=prefill|decode}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
MY_IP="${MY_IP:?MY_IP}"
PORT="${PORT:?PORT}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
TP="${TP:-4}"
BASE_GPU="${BASE_GPU:?BASE_GPU}"
CONC="${CONC:-64}"
MEMFRAC="${MEMFRAC:-0.85}"   # TP4 weights ~210GB/GPU -> need >=0.76
CTX="${CTX:-9472}"
NIC_DEV="${NIC_DEV:?NIC_DEV=mlx5_0|ionic_0}"
GID="${GID:?GID (mlx5=3, ionic=1)}"
IFACE="${IFACE:-ens3}"       # gloo/bootstrap iface (mlx5=ens3; ionic has no IP so still ens3)
LOG="${LOG:-/tmp/tp4rdma_${ROLE}_${PORT}.log}"

# --- RDMA loopback (NO hip transport) ---
unset MC_ENABLE_HIP_TRANSPORT
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS="$NIC_DEV"
export MC_GID_INDEX="$GID"
export RDMAV_FORK_SAFE=1     # ionic needs this; harmless for mlx5
export SGLANG_LOCAL_IP_NIC="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
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
  --disaggregation-ib-device "$NIC_DEV"
  --watchdog-timeout 3600 --chunked-prefill-size 8192
)
if [ "$ROLE" = "prefill" ]; then
  COMMON+=(--disaggregation-bootstrap-port "$BOOTSTRAP")
fi

python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1
