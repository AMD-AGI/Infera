#!/bin/bash
# GLM-5.2-MXFP4 SINGLE-NODE mix server with optional DP-attention (DPA) and MTP.
# Used to reproduce + fix the DSA indexer topk crash when DPA and MTP are fused.
set -u
MY_IP="${MY_IP:?MY_IP}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DPA="${DPA:-1}"
MTP="${MTP:-1}"
CTX="${CTX:-32768}"
MAX_RUNNING="${MAX_RUNNING:-256}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
GMU="${GMU:-0.82}"
NIC="${NIC:-ens3}"
LOG="${LOG:-/tmp/mix_${PORT}.log}"
mkdir -p "$(dirname "$LOG")"

ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"

# ---- GLM-5.2 DSA-ROCm recipe (same as the proven pd_leg_spur.sh) ----
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
fi

MTP_ARGS=()
if [ "$MTP" = "1" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}")
fi

echo "[glm52-mix] ip=$MY_IP dpa=$DPA mtp=$MTP gmu=$GMU chunk=$CHUNK ctx=$CTX -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${DP_ARGS[@]}" "${MTP_ARGS[@]}" > "$LOG" 2>&1
