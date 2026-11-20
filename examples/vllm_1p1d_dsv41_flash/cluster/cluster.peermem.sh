#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE (or export the same names). Nothing else in the kit is site-specific.
# ============================================================================
#
# Wrapper B — amdgpu 6.16 WITH a GPU peer-memory module loaded (ib_peer_mem / amdp2p).
#
# preflight_rdma.sh says "MR_MODE=peermem" on these nodes. Bare ibv_reg_mr hands the NIC the GPU
# pages through peer memory, which KFD charges once on 6.16; dma-buf (charged twice there) is off.
# Same deployment shape as cluster.gem.sh — only section 3 differs.
#
# Usage:  bash cluster/cluster.peermem.sh up | smoke | accuracy | golden | block | agentx CONC... | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Nodes
export PREFILL_NODE="${PREFILL_NODE:-<prefill-host>}"
export DECODE_NODE="${DECODE_NODE:-$PREFILL_NODE}"
export PREFILL_IP="${PREFILL_IP:-<prefill-data-plane-ip>}"
export DECODE_IP="${DECODE_IP:-$PREFILL_IP}"
export KIT_DIR="$KIT"
export PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
export DECODE_GPUS="${DECODE_GPUS:-4,5}"

# 2. Image, weights, logs
export INFERA_IMAGE="${INFERA_IMAGE:-<image from build_image.sh>}"
export MODEL_MOUNT="${MODEL_MOUNT:-<host-dir-holding-the-weights>}"
export MODEL="${MODEL:-$MODEL_MOUNT/DeepSeek-V4.1-Flash}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export LOG_DIR="${LOG_DIR:-/tmp/dsv41_pd}"

# 3. Transport — peer-memory mode
export MR_MODE=peermem
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<ionic_0,...,ionic_7>}"
export RDMA_GPU_NICS="${RDMA_GPU_NICS:-$RDMA_IB_DEVICES}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"

# 4. Deployment shape
export UMBP="${UMBP:-1}"
export UMBP_GB="${UMBP_GB:-449}"
export ETCD_PORT="${ETCD_PORT:-2379}" ROUTER_PORT="${ROUTER_PORT:-8100}"

# 5. AgentX client
export INFERENCEX_DIR="${INFERENCEX_DIR:-<SemiAnalysisAI/InferenceX checkout>}"
export AGENTX_HF_HOME="${AGENTX_HF_HOME:-<HF cache holding semianalysisai/cc-traces-weka-062126>}"

exec bash "$KIT/cluster/run.sh" "$@"
