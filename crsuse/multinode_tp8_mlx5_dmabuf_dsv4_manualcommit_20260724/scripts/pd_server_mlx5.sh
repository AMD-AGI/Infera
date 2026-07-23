#!/bin/bash
# DSv4-Pro 1P1D PD leg (prefill|decode) over NATIVE sglang.launch_server + mooncake,
# FORCED onto the single mlx5 NIC (all 8 GPUs' KV transfer). Run INSIDE the container.
#
# Why mlx5 + dmabuf: mlx5 HAS ODP -> ibv_reg_dmabuf_mr dynamic-attach -> KV pool NOT
# pinned/doubled (the whole point). Image has mooncake rebuilt with USE_HIP_DMABUF.
#
# Usage (inside container):
#   ROLE=prefill|decode MY_IP=<this mlx5 IP> P_IP=<prefill mlx5 IP> \
#   MODEL=/shared_nfs/.../DeepSeek-V4-Pro bash pd_server_mlx5.sh
set -x
ROLE="${ROLE:?ROLE=prefill|decode}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
MY_IP="${MY_IP:?MY_IP=this node mlx5 IP}"
P_IP="${P_IP:?P_IP=prefill mlx5 IP (bootstrap host)}"
PORT="${PORT:-30000}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
TP="${TP:-8}"
CONC="${CONC:-128}"
NIC="${NIC:-ens3}"                 # mlx5 netdev (the only IP NIC)
GID="${GID:-3}"                    # mlx5 RoCEv2 routable GID index
CHUNK="${CHUNK:-8192}"
CTX="${CTX:-9472}"
LOG="${LOG:-/tmp/pd_${ROLE}_${PORT}.log}"

# prefill needs headroom (DPA high-conc); decode can go higher
if [ "$ROLE" = "prefill" ]; then MEMFRAC="${MEMFRAC:-0.85}"; else MEMFRAC="${MEMFRAC:-0.90}"; fi

# ---- force mlx5 for RDMA + gloo; keep KV OFF RCCL ----
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX="$GID"
# force mooncake to the single mlx5 NIC (don't let it grab the 8 ionic rails)
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS="mlx5_0"
# cross-node PD MUST stay RDMA (do NOT enable hip transport)
unset MC_ENABLE_HIP_TRANSPORT
# generous cross-node bootstrap timeouts (cold NFS weights)
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# ---- R4 DSv4 perf env (verbatim from proven recipe) ----
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

COMMON=(
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4
  --cuda-graph-max-bs 512 --disable-radix-cache --page-size 256
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --max-running-requests "$CONC"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device mlx5_0
  --watchdog-timeout 3600 --chunked-prefill-size "$CHUNK"
)
if [ "$ROLE" = "prefill" ]; then
  COMMON+=(--disaggregation-bootstrap-port "$BOOTSTRAP")
fi

python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1
