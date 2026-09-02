#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else needs changing.
# ============================================================================
#
# GLM-5.3 (big) 1P1D on a SINGLE 8-GPU node: TP4 prefill on cards 0-3, TP4
# decode on cards 4-7, KV moved between them over mooncake.
#
# UNVALIDATED. The two-node shape is validated for GLM-5.2; this one is not, and
# its load-bearing unknown is whether mooncake moves KV between two legs on the
# SAME host, and at what speed. Read the README's "single-node unknown" section
# before trusting any number this produces. `smoke` checks MC_FORCE_TCP and
# GID is NULL in both leg logs precisely because a silent fall back to TCP is
# the failure mode here.
#
# It drives the GLM-5.2 kit's engine scripts unchanged -- GLM-5.3 (big) is the
# same architecture, so there is nothing to fork. See the README.
#
# Usage:  bash cluster.singlenode.sh up | smoke | bench [conc...] | down
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The kit whose engine/ and common.sh actually run. Do not point this at a copy.
export KIT_DIR="${KIT_DIR:-$(cd "$HERE/../sglang_1p1d_glm5.2" && pwd)}"

# ---------------------------------------------------------------------------
# 1. Node -- the same host twice
# ---------------------------------------------------------------------------
# Both legs live here, so PREFILL_* and DECODE_* name one machine. up.sh reaches
# each "node" through $SSH_CMD; pointing both at this host is what makes the
# single-node shape work with no change to the engine scripts.
export PREFILL_NODE="${PREFILL_NODE:-$(hostname -s)}"
export DECODE_NODE="$PREFILL_NODE"
# DATA-PLANE IP, not the management NIC -- the legs advertise it to each other.
export PREFILL_IP="${PREFILL_IP:-<this-node-data-plane-ip>}"
export DECODE_IP="$PREFILL_IP"

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
# The STOCK infera sglang image (deploy/docker/Dockerfile.sglang). GLM-5.3 big
# needs no source overlay -- that is only for the Flash family.
export IMAGE="${IMAGE:-<infera-sglang-image>}"
# GLM-5.3-MXFP4 or GLM-5.3. Resolve symlinks: where this path crosses an NFS
# mount boundary, bind-mounting the symlink's parent gives the container an
# empty directory, and the failure surfaces much later as an unrelated error.
export MODEL="${MODEL:-<weights-dir>}"
export MODEL_MOUNT="${MODEL_MOUNT:-$(dirname "$MODEL")}"
export SERVED="${SERVED:-glm-5.3-mxfp4}"

# ---------------------------------------------------------------------------
# 3. Transport
# ---------------------------------------------------------------------------
# From `preflight_rdma.sh mode`, run on THIS node. Do not guess these.
# MC_GID_INDEX is per node: the link-local fe80:: GID is never the answer.
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<ionic_0,ionic_1,...>}"
export MC_GID_INDEX="${MC_GID_INDEX:-<index-from-preflight>}"
# Leave MC_MS_FILTERS unset in mode A (peer-mem present). It is required only in
# the dma-buf mode, where KV must be pinned to one ODP-capable card.
# export MC_MS_FILTERS="ionic_0"

# ---------------------------------------------------------------------------
# 4. Shape
# ---------------------------------------------------------------------------
export TP="${TP:-4}"
export PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
export DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"

# Ports must not collide -- both legs share one host's network namespace. CHECK
# with `ss -lnt`; on a shared node the obvious ones are often taken.
export PREFILL_PORT="${PREFILL_PORT:-30000}"
export DECODE_PORT="${DECODE_PORT:-30001}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
export ROUTER_PORT="${ROUTER_PORT:-8100}"
export ETCD_PORT="${ETCD_PORT:-12379}"

# ---------------------------------------------------------------------------
# 5. Features -- see the README before changing these two
# ---------------------------------------------------------------------------
# MTP off: upstream's GLM-5.3 cookbook disables EAGLE on AMD while the vendor
# model card runs it at 3 steps. Off avoids an unvalidated variable on a shape
# that is itself unvalidated. Turn it on deliberately, as its own round.
export MTP="${MTP:-0}"
export DPA="${DPA:-1}"          # decode-side DP-attention, as in the GLM-5.2 kit
export KVAWARE="${KVAWARE:-1}"
export PREFILL_KVD="${PREFILL_KVD:-0}"
export DECODE_KVD="${DECODE_KVD:-0}"
# Insurance, not a fix -- this checkpoint's shared experts are themselves MXFP4.
# Kept on because upstream #25261 shows the mismatch failing SILENTLY with wrong
# output when shapes happen to line up. See the README.
export EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:---disable-shared-experts-fusion}"

# Prefill wants activation headroom, decode wants KV pool. Do not equalise them.
export GMU_PREFILL="${GMU_PREFILL:-0.70}"
export GMU_DECODE="${GMU_DECODE:-0.85}"

for v in PREFILL_IP IMAGE MODEL RDMA_IB_DEVICES MC_GID_INDEX; do
  case "${!v}" in "<"*) echo "edit $(basename "$0"): $v is still a placeholder" >&2; exit 2;; esac
done

exec bash "$KIT_DIR/engine/${1:?usage: $(basename "$0") up|smoke|bench|down}.sh" "${@:2}"
