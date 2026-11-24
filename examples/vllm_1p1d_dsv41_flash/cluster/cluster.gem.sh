#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE (or export the same names). Nothing else in the kit is site-specific.
# ============================================================================
#
# Wrapper A — amdgpu >= 6.19, or ANY amdgpu without a GPU peer-memory module.
#
# preflight_rdma.sh says "MR_MODE=gem" on these nodes. Their KFD charges a VRAM buffer pinned
# for RDMA twice — through dma-buf on 6.16+, through peer memory too on 6.19 — so a large KV
# cache cannot be registered. The legs instead allocate the KV cache from amdgpu GEM BOs
# (charged once; infera.engine.vllm.gem_mooncake + the image's libhip_gem_alloc.so) and Mooncake
# registers them by dma-buf. Validated on AUS MI355X (amdgpu 6.19.14, 8x ionic 400G).
#
# Usage:  bash cluster/cluster.gem.sh up | smoke | accuracy | golden | block | agentx CONC... | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Nodes. A single-node pair sets both to the same host and splits the GPUs.
#    Run the wrapper ON the prefill node (e.g. under `srun`), or set SSH_CMD.
# ---------------------------------------------------------------------------
export PREFILL_NODE="${PREFILL_NODE:-<prefill-host>}"
export DECODE_NODE="${DECODE_NODE:-$PREFILL_NODE}"
export PREFILL_IP="${PREFILL_IP:-<prefill-data-plane-ip>}"
export DECODE_IP="${DECODE_IP:-$PREFILL_IP}"
export KIT_DIR="$KIT"                    # must be the SAME path on both nodes
# GPUs per leg; the DP size of each leg is the number listed. The c384 shape is DP2 + DP2.
# On a shared node, split them across the two sockets: each leg's Engram tables live in host
# memory, and each GPU uses the NIC on its own PCIe switch.
export PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
export DECODE_GPUS="${DECODE_GPUS:-4,5}"

# ---------------------------------------------------------------------------
# 2. Image, weights, logs
# ---------------------------------------------------------------------------
export INFERA_IMAGE="${INFERA_IMAGE:-<image from build_image.sh>}"
export MODEL_MOUNT="${MODEL_MOUNT:-<host-dir-holding-the-weights>}"
export MODEL="${MODEL:-$MODEL_MOUNT/DeepSeek-V4.1-Flash}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export LOG_DIR="${LOG_DIR:-/tmp/dsv41_pd}"   # host path; shared storage if the legs are on two nodes

# ---------------------------------------------------------------------------
# 3. Transport — GEM mode. Copy these from preflight_rdma.sh, do not guess.
# ---------------------------------------------------------------------------
export MR_MODE=gem
# Every PORT_ACTIVE RDMA device, and the NIC for each GPU index (position k = GPU k).
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<ionic_0,...,ionic_7>}"
export RDMA_GPU_NICS="${RDMA_GPU_NICS:-$RDMA_IB_DEVICES}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"     # routable RoCE v2 GID; per node, check both

# ---------------------------------------------------------------------------
# 4. Deployment shape (defaults = the recommended configuration, see README)
# ---------------------------------------------------------------------------
export UMBP="${UMBP:-1}"                     # host-DRAM KV tier on prefill
export UMBP_GB="${UMBP_GB:-449}"             # absolute size; must fit the node's free RAM
# UMBP's standalone server maps the prefill KV cache with hipIpcOpenMemHandle. HIP itself has no IPC
# handle for GEM memory; the image's GEM allocator supplies one (UMBP_GEM_IPC=1, default), so the
# prefill stays on GEM at 0.9 and the server preloads the allocator. UMBP_GEM_IPC=0 falls back to
# ordinary KFD memory registered by dma-buf, which KFD charges twice on these drivers: 0.82 leaves
# room for the second charge (~38 GB of KV per rank instead of ~61 GB at 0.9).
# UMBP_MODE=embedded keeps a DRAM pool inside each prefill DP rank (no IPC at all).
export UMBP_MODE="${UMBP_MODE:-standalone}"
if [ "$UMBP" = "1" ] && [ "$UMBP_MODE" = "standalone" ] && [ "${UMBP_GEM_IPC:-1}" = "0" ]; then
  export PREFILL_MR_MODE="${PREFILL_MR_MODE:-dmabuf}" PREFILL_GMU="${PREFILL_GMU:-0.82}"
fi
export ETCD_PORT="${ETCD_PORT:-2379}" ROUTER_PORT="${ROUTER_PORT:-8100}"

# ---------------------------------------------------------------------------
# 5. AgentX client (agentx subcommand only)
# ---------------------------------------------------------------------------
export INFERENCEX_DIR="${INFERENCEX_DIR:-<SemiAnalysisAI/InferenceX checkout>}"
export AGENTX_HF_HOME="${AGENTX_HF_HOME:-<HF cache holding semianalysisai/cc-traces-weka-062126>}"

exec bash "$KIT/cluster/run.sh" "$@"
