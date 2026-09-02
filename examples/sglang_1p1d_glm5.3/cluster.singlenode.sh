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
#
# TWO names for one image, and both are needed. preflight_rdma.sh reads IMAGE;
# engine/up.sh and common.sh require INFERA_IMAGE. Exporting only IMAGE makes
# `up` die at its first require_env, before a single container is started.
export IMAGE="${IMAGE:-<infera-sglang-image>}"
export INFERA_IMAGE="${INFERA_IMAGE:-$IMAGE}"
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
# SINGLE-NODE RULE, and it differs from the two-node one: pin ONE device, the
# same on both legs. preflight_rdma.sh states it directly under mode A --
# "all active rails (cross-node; mooncake pairs by GID subnet). Single-node
# loopback: pin ONE device on both legs instead." Listing all rails here is a
# cross-node recipe applied to a shape that is not cross-node.
#
# This matters even when the hip transport is enabled below: if hip fails to
# install, the fallback is loopback RDMA, and this is what decides whether that
# fallback is correctly configured or not.
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<one-ionic-device, e.g. ionic_0>}"
export MC_GID_INDEX="${MC_GID_INDEX:-<index-from-preflight>}"
# preflight_rdma.sh asks for this in ALL THREE modes, and engine/leg.sh honours
# it only when it is passed in ([ "${RDMAV_FORK_SAFE:-0}" = "1" ]). Unset here it
# never reaches the engine, so the preflight's own recommendation would be
# silently dropped.
export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
# Leave MC_MS_FILTERS unset in mode A (peer-mem present). It is required only in
# the dma-buf mode, where KV must be pinned to one ODP-capable card.
# export MC_MS_FILTERS="ionic_0"

# THE flag that decides how single-node KV actually moves, and the only place it
# is turned on. engine/leg.sh defaults it to 1 (hip transport OFF), which is
# right for the two-node shape and wrong here: with hip absent the local segment
# advertises "rdma" only, and KV between two legs on ONE host takes LOOPBACK
# RDMA. That path works, raises nothing, and is the silent-slow case -- the
# README's "it fails loudly" only holds when hip is installed.
#
# With hip installed, MultiTransport::selectTransport routes KV by fixed
# priority (hip 4 > cxl 3 > rdma 2 > tcp 1), so hip wins: hipIpcGetMemHandle on
# the exporter, hipIpcOpenMemHandle on the importer, GPU-to-GPU over XGMI with
# no NIC in the path. That is what the pinned mooncake commit 01d1eb2a exists
# for. Measured on this hardware: HIP IPC works across DISJOINT
# HIP_VISIBLE_DEVICES -- an importer that cannot see the exporter's physical GPU
# mapped its memory and read back the correct bytes.
#
# REQUIRED CHECK after bring-up: `HIP transport installed for intra-node GPU
# P2P` must appear in BOTH leg logs, and `hipIpcOpenMemHandle failed` in
# neither. The MC_FORCE_TCP and GID-is-NULL counters do NOT cover this -- there
# is no log line for a TCP fallback, and a same-host hip transfer never touches
# a GID.
export MC_DISABLE_HIP_TRANSPORT="${MC_DISABLE_HIP_TRANSPORT:-0}"

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

# The KV-event ports are PER LEG, and on this shape they must differ. In the
# two-node case both legs take engine/leg.sh's defaults (5557/8801) and never
# meet; here they share one network namespace, so the second leg's bind fails
# and the leg never serves -- the same "port_base at N is not available" that
# common.sh's reap() warns about across restarts, happening across legs instead.
# engine/up.sh forwards these per leg.
export PREFILL_KV_PUB_PORT="${PREFILL_KV_PUB_PORT:-5557}"
export PREFILL_KV_SNAP_PORT="${PREFILL_KV_SNAP_PORT:-8801}"
export DECODE_KV_PUB_PORT="${DECODE_KV_PUB_PORT:-5558}"
export DECODE_KV_SNAP_PORT="${DECODE_KV_SNAP_PORT:-8802}"

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

for v in PREFILL_IP IMAGE INFERA_IMAGE MODEL RDMA_IB_DEVICES MC_GID_INDEX; do
  case "${!v}" in "<"*) echo "edit $(basename "$0"): $v is still a placeholder" >&2; exit 2;; esac
done

exec bash "$KIT_DIR/engine/${1:?usage: $(basename "$0") up|smoke|bench|down}.sh" "${@:2}"
