#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else in this kit needs changing for your cluster.
# ============================================================================
#
# Wrapper A — MULTI-RAIL RDMA WITH A PEER-MEMORY MODULE.
#
# Use this when `python -m infera.tools.preflight.mooncake_mode` reports
# **mode A viable** on your nodes, i.e.:
#
#     peermem : present
#     mode A  : VIABLE   bare ibv_reg_mr + peer-mem (default, no-pin, every rail)
#
# In this mode registration hands the NIC the GPU pages directly: nothing is
# pinned, the KV pool is not duplicated, and every active rail can carry KV.
# dma-buf is therefore switched OFF — it would be strictly worse here.
#
# If preflight instead reports "no peer-mem module loaded", use
# cluster.dmabuf.sh. See cluster/README.md for the full mapping.
#
# Usage:  bash cluster/cluster.peermem.sh up | smoke | bench [conc...] | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
# Each node's SSH-reachable name and its DATA-PLANE IP. Do NOT use the
# management/public NIC — the legs advertise these to each other for the KV
# handoff. See cluster/README.md section 1.
export PREFILL_NODE="<prefill-host>"
export DECODE_NODE="<decode-host>"
export PREFILL_IP="<prefill-data-plane-ip>"
export DECODE_IP="<decode-data-plane-ip>"

# Must be the SAME path on both nodes — up.sh runs engine/leg.sh from here.
export KIT_DIR="$KIT"

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
# Requires an infera-sglang build NEWER than 0.2.0. The default below is a placeholder.
export INFERA_IMAGE="${INFERA_IMAGE:-<infera-sglang-image>}"

# MODEL_MOUNT is bind-mounted into the container; MODEL must live under it.
# Prefer LOCAL storage on both nodes — a slow mount blows the ready timeout
# (see cluster/README.md section 2).
export MODEL_MOUNT="<host-dir-holding-the-weights>"
export MODEL="$MODEL_MOUNT/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# If your image injects a host RDMA provider library at entrypoint, set all three: the
# host library (the SYMLINK, so differing per-node builds resolve), the in-container path
# THAT image reads, and the entrypoint. Any one alone silently does nothing — see §2.
# export HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/lib<provider>.so.1
# export HOST_RDMA_MOUNT=/host-<provider>/lib<provider>.so ENTRYPOINT_KEEP=1

# ---------------------------------------------------------------------------
# 3. Transport — mode A (peer-mem, every rail)
# ---------------------------------------------------------------------------
# Copy these two values straight out of the preflight report's mode-A block.
# RDMA_IB_DEVICES is every ACTIVE rail — a rail that is physically down must NOT
# be listed, or every transfer targeting it fails. Enumerate per node; see §3.
export RDMA_IB_DEVICES="<ionic_0,ionic_1,...>"

# MC_GID_INDEX: the RoCEv2 GID index. It is NODE-dependent, not cluster-wide —
# if your two nodes disagree, set it per node rather than here. The link-local
# fe80:: GID (usually index 0) is NOT a fallback. Details: cluster/README.md §3.
export MC_GID_INDEX="<gid-index-from-preflight>"

# dma-buf OFF: with a peer-mem module loaded, bare ibv_reg_mr is the no-pin path.
export MOONCAKE_DISABLE_HIP_DMABUF=1
# Some providers need fork-safety enabled in libibverbs.
export RDMAV_FORK_SAFE=1
# MC_TE_FILTERS is deliberately unset: every rail is allowed to carry KV.

# ---------------------------------------------------------------------------
# 4. Deployment shape — see the README's "Recommended configuration"
# ---------------------------------------------------------------------------
export TP=8
export CTX=262144
export CHUNK=65536                 # GLOBAL per-step prefill budget (see engine/leg.sh)

export PREFILL_DPA=0               # prefill DP-attention: 0 = pure TP (recommended)
export DECODE_DPA=1                # decode DP-attention:  1 = dp8
export PREFILL_MTP=0               # MTP on prefill: leave off
export DECODE_MTP=1                # MTP on decode: EAGLE speculative decoding

export ROUTER_POLICY=kv-aware      # or round-robin — read the README first, the two
                                   # are coupled to GMU_PREFILL
export KVAWARE=1
export PREFILL_KVD=1               # kvd L2/L3 on the prefill leg
export DECODE_KVD=0                # off on decode by design

export GMU_PREFILL=0.70
export GMU_DECODE=0.85

exec bash "$KIT/engine/${1:-up}.sh" "${@:2}"
