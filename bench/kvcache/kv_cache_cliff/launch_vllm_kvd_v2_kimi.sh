#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Launch Kimi K2.5 MXFP4 TP=4 (or TP=8) with our v2 chunked-fusion
# connector for the KV-cliff bench. Kimi is the MLA model where the
# external read path actually fires under reasonable concurrency
# because per-token KV is ~68 KB (61 layers × 576 elements × 2 B)
# and prefill is comparatively slow (~13 K tok/s aggregate on TP=4
# MI355X), so NVMe reload beats recompute at the eviction point.
#
# Designed to run INSIDE the jj_vllm_gptoss container.
#
# Usage:
#   # 1. Start kvd daemon on /mnt/nvme8:
#   PYTHONPATH=$INFERA_ROOT \
#   python -m infera.kvd --socket /tmp/kvd-kimi.sock \
#       --max-bytes $((4 << 30)) \
#       --long-path /mnt/nvme8/kvd-kimi-long --long-bytes $((100 << 30)) \
#       --spillover-path /mnt/nvme8/kvd-kimi-short --spillover-bytes $((100 << 30)) &
#
#   # 2. Launch vLLM. Default TP=4 uses GPUs 2,3,5,6 (skip 4 which
#   #    is often busy on a shared node).
#   bash $INFERA_ROOT/bench/kvcache/kv_cache_cliff/launch_vllm_kvd_v2_kimi.sh
#
# Output: server listens on PORT (default 8803).

set -uo pipefail

# Repo root, auto-derived from this script's location
# (bench/kvcache/kv_cache_cliff/ -> 3 levels up). Override with
# INFERA_ROOT for a relocated checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERA_ROOT="${INFERA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

GPU_IDX=${GPU_IDX:-2,3,5,6}
TP=${TP:-4}
PORT=${PORT:-8803}
MODEL=${MODEL:-/PATH/TO/amd-Kimi-K2.5-MXFP4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# Cap L1 below the concurrent KV working set so eviction → kvd reads
# actually triggers. 26 GiB is the validated operating point from
# project_kimi_cachetier_operating_point memory.
KV_CACHE_BYTES=${KV_CACHE_BYTES:-$((26 << 30))}
KVD_SOCKET=${KVD_SOCKET:-/tmp/kvd-kimi.sock}
HIPFILE_LONG=${HIPFILE_LONG:-/mnt/nvme8/kvd-kimi-long}
HIPFILE_SHORT=${HIPFILE_SHORT:-/mnt/nvme8/kvd-kimi-short}
LOG=${LOG:-/tmp/vllm-kimi.log}

mkdir -p "$HIPFILE_LONG" "$HIPFILE_SHORT"

# ROCm env — same as the gpt-oss bench so the only variable
# between arms is the connector attachment.
export HIP_VISIBLE_DEVICES="$GPU_IDX"
unset ROCR_VISIBLE_DEVICES
export AMDGCN_USE_BUFFER_OPS=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=${VLLM_ROCM_USE_AITER_FP4_ASM_GEMM:-1}
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_RPC_TIMEOUT=1800000
export PYTHONHASHSEED=0

# Connector wiring. INFERA_KVD_CHUNK_TOKENS=512 → N=32 pages
# per chunk for block_size=16 (Kimi default).
export INFERA_KVD_SOCKET="$KVD_SOCKET"
export INFERA_KVD_HIPFILE_ROOTS="long=${HIPFILE_LONG},short=${HIPFILE_SHORT}"
export INFERA_KVD_CHUNK_TOKENS=${INFERA_KVD_CHUNK_TOKENS:-512}

export PYTHONPATH="${INFERA_ROOT}:${PYTHONPATH:-}"

KV_TRANSFER='{
  "kv_connector": "InferaKvdConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "infera.engine.vllm.kvd_connector"
}'

echo "[launch] GPU=$GPU_IDX TP=$TP PORT=$PORT MODEL=$MODEL" | tee "$LOG"
echo "[launch] kvd=$KVD_SOCKET kv_cache=$KV_CACHE_BYTES" | tee -a "$LOG"

# IMPORTANT: cd to /tmp so sys.path[0] is /tmp (not /app, which is the
# vllm container's default WORKDIR). /app has a `vllm/` directory
# whose presence (without an __init__.py) makes `import vllm` resolve
# as an empty namespace package, shadowing the editable install. The
# editable-finder install then runs second but the namespace
# package's path[] doesn't include /app/vllm/vllm so attribute lookups
# (SamplingParams, etc.) miss. /tmp has no `vllm/` collision so the
# finder wins and `vllm.__file__` correctly resolves to
# /app/vllm/vllm/__init__.py.
cd /tmp

exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.5 \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
  --trust-remote-code \
  --kv-transfer-config "$KV_TRANSFER" \
  2>&1 | tee -a "$LOG"
