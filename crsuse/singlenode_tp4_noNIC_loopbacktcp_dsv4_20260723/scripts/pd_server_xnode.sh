#!/bin/bash
# ============================================================================
# DSv4-Pro CROSS-NODE 1P1D PD server (spur). prefill on node A, decode on node B.
# TP8 per role (full node each). KV over ionic RoCE via mooncake (real RDMA).
# = legacy pd_server.sh wiring, adapted to spur:
#     - control/bootstrap plane = ens3 (real IPs), data plane = ionic RoCE (GID1)
#     - NO ionic injection needed (mori-0615 image already exposes 9 RDMA ports)
#     - backend = mooncake (not mooncake_tcp; we have real RDMA here)
#
# Usage (inside container on the respective node):
#   ROLE=prefill P_IP=<A_ens3> D_IP=<B_ens3> bash pd_server_xnode.sh
#   ROLE=decode  P_IP=<A_ens3> D_IP=<B_ens3> bash pd_server_xnode.sh
# ============================================================================
set -x
ROLE="${ROLE:?ROLE=prefill|decode}"
BACKEND="${BACKEND:-mooncake}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
BOOTSTRAP="${BOOTSTRAP:-8998}"
P_IP="${P_IP:?P_IP=prefill node ens3 IP}"
D_IP="${D_IP:?D_IP=decode node ens3 IP}"
ISL="${ISL:-8192}"
CTX="${CTX:-9472}"
CONC="${CONC:-32}"
IB_DEV="${IB_DEV:-ionic_0}"          # RDMA device for KV transfer
GID="${GID:-1}"                      # RoCEv2 global GID (idx0=fe80 link-local, crashes)
NIC="${NIC:-ens3}"                   # control-plane netdev (bootstrap/TCP)
WORK="${WORK:-/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg}"
LOG="${LOG:-$WORK/xnode_${ROLE}.log}"
mkdir -p "$(dirname "$LOG")"

if [ "$ROLE" = "prefill" ]; then MY_IP="$P_IP"; MEMFRAC="${MEMFRAC:-0.85}"; else MY_IP="$D_IP"; MEMFRAC="${MEMFRAC:-0.90}"; fi

# ---- control-plane / RDMA network env ----
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC" MORI_SOCKET_IFNAME="$NIC"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export MORI_GPU_ARCHS=gfx950 NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX="$GID"

# ---- R4 perf env (legacy manifest full set, verbatim) ----
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

COMMON=(
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4
  --cuda-graph-max-bs 128 --disable-radix-cache --page-size 256
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX"
  --max-running-requests "$CONC"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend "$BACKEND"
  --disaggregation-ib-device "$IB_DEV"
  --watchdog-timeout 3600
  --chunked-prefill-size "$ISL"
)
if [ "$ROLE" = "prefill" ]; then
  COMMON+=(--disaggregation-bootstrap-port "$BOOTSTRAP")
fi

python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1
