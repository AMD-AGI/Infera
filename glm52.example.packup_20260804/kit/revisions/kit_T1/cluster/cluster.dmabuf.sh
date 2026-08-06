#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else in this kit needs changing for your cluster.
# ============================================================================
#
# Wrapper B — NO PEER-MEMORY MODULE, GPUDirect VIA dma-buf ON AN ODP NIC.
#
# Use this when `python -m infera.tools.preflight.mooncake_mode` reports
# **mode B viable** on your nodes, i.e.:
#
#     peermem : absent
#     mode B  : VIABLE   ibv_reg_dmabuf_mr (GPUDirect dma-buf) on the ODP NIC (no-pin)
#
# Without a peer-mem module, dma-buf is the only GPUDirect path — and it is only
# SAFE on a NIC with ODP (on-demand paging). On a NIC without ODP the driver PINS
# the whole registered region, duplicating the KV pool in VRAM until a large pool
# exhausts a KFD resource and the process dies. That is why KV is locked to the
# ODP NIC below and not left to auto-discovery.
#
# Expect preflight to also print a `perf-regression` warning here if the ODP NIC
# is slower or fewer than the node's fastest rails. That is the real cost of
# no-pin dma-buf without peer-mem, and it is a fabric fact, not a misconfiguration.
#
# So: the one-NIC restriction is a FALLBACK, not a target. Mode A (peer-mem) uses
# every rail and pins nothing; this mode exists for nodes where no peer-memory
# module is loaded and loading one is not an option.
#
# If preflight instead reports "peer-mem present", use cluster.peermem.sh.
# See cluster/README.md for the full mapping.
#
# Usage:  bash cluster/cluster.dmabuf.sh up | smoke | bench [conc...] | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
# SSH-reachable names for the two nodes, and each one's DATA-PLANE IP — the
# address on the KV network that the peer node and the router can reach. Do NOT
# use the management/public NIC: the legs advertise these to each other.
#
# On a scheduler that blocks ssh to compute nodes, set SSH_CMD below to whatever
# does run a command on a node (see cluster/README.md).
export PREFILL_NODE="<prefill-host>"
export DECODE_NODE="<decode-host>"
export PREFILL_IP="<prefill-data-plane-ip>"
export DECODE_IP="<decode-data-plane-ip>"

# This kit must live at the SAME path on both nodes (a shared filesystem, or an
# identical copy). up.sh runs engine/leg.sh from here on each node.
export KIT_DIR="$KIT"

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
# Placeholder tag — replace with the released infera-sglang image.
export INFERA_IMAGE="${INFERA_IMAGE:-inferaimage/infera-sglang:0.2.0}"

# MODEL_MOUNT is bind-mounted into the container; MODEL must live under it.
# Weights on shared storage work but load slowly — the checkpoint is ~400 GB and
# both legs read it at once. The ready timeout in engine/leg.sh is sized for that.
export MODEL_MOUNT="<host-dir-holding-the-weights>"
export MODEL="$MODEL_MOUNT/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# ---------------------------------------------------------------------------
# 3. Transport — mode B (dma-buf on the ODP NIC)
# ---------------------------------------------------------------------------
# Copy these values straight out of the preflight report's mode-B block. It
# names the exact NIC and GID it wants, having already checked ODP per card and
# confirmed the image's mooncake engine really has ibv_reg_dmabuf_mr compiled in.
#
# One device, not a list. Both legs MUST name the SAME one — preflight breaks
# ties by name for exactly this reason.
export RDMA_IB_DEVICES="<odp-nic-from-preflight>"          # e.g. mlx5_0

# Lock mooncake to that device. With auto-discovery on, it would find the
# non-ODP rails and register KV on one of them, which pins and doubles the pool.
export MC_MS_AUTO_DISC=0
export MC_MS_FILTERS="$RDMA_IB_DEVICES"

# The RoCEv2 routable GID index for that NIC. Read it from preflight; the
# link-local fe80:: GID is not routable across the fabric and is never the answer.
export MC_GID_INDEX="<gid-index-from-preflight>"

# dma-buf ON — the whole point of this mode.
export MOONCAKE_DISABLE_HIP_DMABUF=0

# Only needed when the node ALSO has non-ODP rails present. Uncomment if
# preflight lists any.
# export RDMAV_FORK_SAFE=1

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
