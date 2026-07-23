#!/bin/bash
# REPRO: force KV transfer over ionic (NO ODP) with the dmabuf-enabled image.
# Expectation: mooncake registers the KV pool via ibv_reg_dmabuf_mr on ionic; with
# no ODP the amdgpu driver must PIN the whole pool -> a 2nd VRAM copy (doubling) or
# a KFD/driver resource exhaustion -> OOM / HIP-209 -> engine crash at KV setup.
#
# Single node, single decode leg is enough: the KV pool registration happens at
# server init (before any peer), so the failure triggers standalone.
# Run INSIDE the container.
set -x
MODEL="${MODEL:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro}"
MY_IP="${MY_IP:?MY_IP}"      # ionic has no IP; use the mlx5/ens3 IP just for the http/bootstrap plane
PORT="${PORT:-30000}"
TP="${TP:-8}"
CONC="${CONC:-128}"
GID="${GID:-1}"             # ionic_0 RoCEv2 global GID index (fc01:... = idx1)
MEMFRAC="${MEMFRAC:-0.90}"  # high -> big KV pool -> pin-doubling definitely OOMs
CTX="${CTX:-9472}"
LOG="${LOG:-/tmp/repro_ionic.log}"

# --- force IONIC (the whole point) ---
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX="$GID"
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS="ionic_0"   # mooncake -> ionic only
export RDMAV_FORK_SAFE=1                            # ionic needs this (else rdma_context setup fails [22])
unset MC_ENABLE_HIP_TRANSPORT

# R4 perf env (same as mlx5 run)
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
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" --attention-backend dsv4 \
  --cuda-graph-max-bs 512 --disable-radix-cache --page-size 256 \
  --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion \
  --mem-fraction-static "$MEMFRAC" --context-length "$CTX" \
  --max-running-requests "$CONC" \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device ionic_0 \
  --watchdog-timeout 3600 --chunked-prefill-size 8192 \
  > "$LOG" 2>&1
