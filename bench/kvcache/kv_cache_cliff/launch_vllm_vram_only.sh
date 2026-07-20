#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch vLLM with gpt-oss-120b MXFP4 TP=1 for the KV-cliff bench's
# Arm A (vram-only baseline). Pins to a single free GPU; prefix
# cache is vLLM's default (VRAM only). No kv_transfer_config.
#
# Designed to run INSIDE the jj_vllm_gptoss container.
# Pre-req: GPU `$GPU_IDX` must be free (other GPUs left alone so the
# user's Qwen vLLM on card3 keeps running).
#
# Usage:
#   bash $INFERA_ROOT/bench/kvcache/kv_cache_cliff/launch_vllm_vram_only.sh
#
# Output: server listens on PORT (default 8802). Tail the log to
# verify it's ready; the bench reads /v1/models to ping-test before
# starting the sweep.

set -uo pipefail

GPU_IDX=${GPU_IDX:-0}
PORT=${PORT:-8802}
MODEL=${MODEL:-/PATH/TO/gpt-oss-120b-w-mxfp4-a-fp8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
LOG=${LOG:-/tmp/vllm-cliff-vram-only.log}

# ROCm env: matches john's run_gpt_oss.sh recipe so gpt-oss MXFP4
# initializes cleanly on MI355X.
export HIP_VISIBLE_DEVICES="$GPU_IDX"
export ROCR_VISIBLE_DEVICES="$GPU_IDX"
export AMDGCN_USE_BUFFER_OPS=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=${VLLM_ROCM_USE_AITER_FP4_ASM_GEMM:-1}
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_RPC_TIMEOUT=1800000
# PYTHONHASHSEED=0 ensures vLLM's block-content hashes are
# deterministic — required for the cliff bench's shared-prefix
# warming pattern to actually share cache across vLLM restarts (see
# memory project_pythonhashseed_required).
export PYTHONHASHSEED=0

echo "[launch] GPU=$GPU_IDX  PORT=$PORT  MODEL=$MODEL  MAX_LEN=$MAX_MODEL_LEN" | tee "$LOG"
echo "[launch] starting vLLM (Arm A: vram-only baseline; no kv_transfer_config)" | tee -a "$LOG"

exec vllm serve "$MODEL" \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  2>&1 | tee -a "$LOG"
