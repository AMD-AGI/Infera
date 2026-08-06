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
# SSH-reachable names for the two nodes, and each one's DATA-PLANE IP — the
# address on the RDMA rail network that the peer node and the router can reach.
# Do NOT use the management/public NIC here: the legs advertise these addresses
# to each other for the KV handoff.
export PREFILL_NODE="<prefill-host>"
export DECODE_NODE="<decode-host>"
export PREFILL_IP="<prefill-data-plane-ip>"
export DECODE_IP="<decode-data-plane-ip>"

# This kit must live at the SAME path on both nodes (a shared filesystem, or an
# identical copy). up.sh ssh's in and runs engine/leg.sh from here.
export KIT_DIR="$KIT"

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
# Placeholder tag — replace with the released infera-sglang image.
export INFERA_IMAGE="${INFERA_IMAGE:-inferaimage/infera-sglang:0.2.0}"

# MODEL_MOUNT is bind-mounted into the container; MODEL must live under it.
# Both nodes need the weights on LOCAL storage. A mount that merely looks shared
# may be an NFS export of the peer's array, which turns an 8-minute weight load
# into a ~95-minute one and blows the ready timeout.
export MODEL_MOUNT="<host-dir-holding-the-weights>"
export MODEL="$MODEL_MOUNT/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# Some images bind-map a host RDMA provider library at entrypoint so the
# container's provider matches the host driver. If yours does, point this at the
# host library and drop the `--entrypoint ''` override:
# export HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/lib<provider>.so
# export ENTRYPOINT_KEEP=1

# ---------------------------------------------------------------------------
# 3. Transport — mode A (peer-mem, every rail)
# ---------------------------------------------------------------------------
# Copy these two values straight out of the preflight report's mode-A block.
#
# RDMA_IB_DEVICES: every ACTIVE rail. Enumerate rather than hardcode — a rail
# that is physically down must not be listed, or every transfer targeting it
# fails. This one-liner lists the active ones on the node you run it on:
#
#     for d in /sys/class/infiniband/*; do
#       grep -q ACTIVE "$d/ports/1/state" && basename "$d"
#     done | paste -sd,
#
export RDMA_IB_DEVICES="<ionic_0,ionic_1,...>"

# MC_GID_INDEX: the RoCEv2 GID index, and it is NODE-DEPENDENT. Nodes of the
# same model routinely expose the routable GID at different indices — one at 1,
# another at 2 — because an empty slot shifts everything after it. Getting it
# wrong fails loudly at init ("GID is NULL, please check your GID index") on
# every DP rank, so it is cheap to catch but expensive to assume.
#
# The link-local fe80:: GID (usually index 0) is NOT a fallback: it is not
# routable across the fabric.
#
# If your two nodes disagree, set MC_GID_INDEX per node instead of here — e.g.
# export it in the per-node environment, or split this wrapper in two.
export MC_GID_INDEX="<gid-index-from-preflight>"

# dma-buf OFF: with a peer-mem module loaded, bare ibv_reg_mr is the no-pin path.
export MOONCAKE_DISABLE_HIP_DMABUF=1
# Some providers need fork-safety enabled in libibverbs.
export RDMAV_FORK_SAFE=1
# MC_MS_FILTERS is deliberately unset: every rail is allowed to carry KV.

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
