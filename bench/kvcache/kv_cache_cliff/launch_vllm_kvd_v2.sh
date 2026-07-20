#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch vLLM with gpt-oss-120b MXFP4 TP=1 + InferaKvdConnector (v2
# chunked-fusion) for the KV-cliff bench's Arm B. Expects a kvd
# daemon already running on `$KVD_SOCKET` with hipfile_roots
# configured.
#
# Designed to run INSIDE the jj_vllm_gptoss container.
#
# Usage:
#   # 1. Start kvd daemon with file tier on /mnt/nvme8:
#   PYTHONPATH=$INFERA_ROOT \
#   python -m infera.kvd --socket /tmp/kvd-cliff.sock \
#       --max-bytes $((4 << 30)) \
#       --long-path /mnt/nvme8/kvd-cliff-long --long-bytes $((80 << 30)) \
#       --spillover-path /mnt/nvme8/kvd-cliff-short --spillover-bytes $((80 << 30)) &
#
#   # 2. Launch vLLM attached to kvd:
#   bash $INFERA_ROOT/bench/kvcache/kv_cache_cliff/launch_vllm_kvd_v2.sh
#
# Output: server listens on PORT (default 8803).

set -uo pipefail

# Repo root, auto-derived from this script's location
# (bench/kvcache/kv_cache_cliff/ -> 3 levels up). Override with
# INFERA_ROOT for a relocated checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERA_ROOT="${INFERA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

GPU_IDX=${GPU_IDX:-0}
PORT=${PORT:-8803}
MODEL=${MODEL:-/PATH/TO/gpt-oss-120b-w-mxfp4-a-fp8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
KVD_SOCKET=${KVD_SOCKET:-/tmp/kvd_pool.sock}
HIPFILE_LONG=${HIPFILE_LONG:-/mnt/nvme8/kvd-cliff-long}
HIPFILE_SHORT=${HIPFILE_SHORT:-/mnt/nvme8/kvd-cliff-short}
LOG=${LOG:-/tmp/vllm-cliff-kvd-v2.log}

mkdir -p "$HIPFILE_LONG" "$HIPFILE_SHORT"

# Same ROCm env as the vram-only arm so the two are apples-to-apples
# on everything except the connector.
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
export PYTHONHASHSEED=0

# Connector config: chunked-fusion is the only encoding (vLLM-9
# dropped v1). Chunk size default 512 tokens. hipfile_roots points
# at /mnt/nvme8 (local NVMe RAID-0); flip to an NFS mount for
# NFS-backed runs.
export INFERA_KVD_SOCKET="$KVD_SOCKET"
export INFERA_KVD_HIPFILE_ROOTS="long=${HIPFILE_LONG},short=${HIPFILE_SHORT}"
export INFERA_KVD_CHUNK_TOKENS=${INFERA_KVD_CHUNK_TOKENS:-512}
# GPU-direct opt-in only when explicitly requested (default POSIX).
# Flip to true for Vast NFS where hipFile beats POSIX 2-5×.
# export INFERA_KVD_GPU_DIRECT=true

# Inject our connector module path for vLLM's KVConnectorFactory.
export PYTHONPATH="${INFERA_ROOT}:${PYTHONPATH:-}"

KV_TRANSFER='{
  "kv_connector": "InferaKvdConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "infera.engine.vllm.kvd_connector"
}'

echo "[launch] GPU=$GPU_IDX  PORT=$PORT  MODEL=$MODEL  KVD=$KVD_SOCKET" | tee "$LOG"
echo "[launch] hipfile_roots: long=$HIPFILE_LONG short=$HIPFILE_SHORT" | tee -a "$LOG"
echo "[launch] starting vLLM (Arm B: chunked-fusion v2 + kvd-on-NVMe)" | tee -a "$LOG"

exec vllm serve "$MODEL" \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --kv-transfer-config "$KV_TRANSFER" \
  2>&1 | tee -a "$LOG"
