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
# Same-host KV moves over HIP IPC across XGMI rather than the NIC, which makes
# this shape's checks different from the two-node one -- read the README's
# single-node section before trusting any number this produces.
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

# The STOCK infera sglang image (deploy/docker/Dockerfile.sglang) -- GLM-5.3 big
# needs no source overlay. TWO names for one image, and both are needed:
# preflight_rdma.sh reads IMAGE, engine/up.sh and common.sh require INFERA_IMAGE.
# Exporting only IMAGE makes `up` die at its first require_env, before any container.
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

# From `preflight_rdma.sh mode`, run on THIS node. Do not guess these; the
# link-local fe80:: GID is never the answer. SINGLE-NODE RULE, differing from the
# two-node one: pin ONE device, the same on both legs. It still matters with hip
# enabled below -- if hip fails to install, the fallback is loopback RDMA.
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<one-ionic-device, e.g. ionic_0>}"
export MC_GID_INDEX="${MC_GID_INDEX:-<index-from-preflight>}"
# preflight_rdma.sh asks for this in ALL THREE modes, and engine/leg.sh honours
# it only when it is passed in ([ "${RDMAV_FORK_SAFE:-0}" = "1" ]). Unset here it
# never reaches the engine, so the preflight's own recommendation would be
# silently dropped.
export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
# Leave MC_TE_FILTERS unset in mode A (peer-mem present). It is required only in
# the dma-buf mode, where KV must be pinned to one ODP-capable card.
# export MC_TE_FILTERS="ionic_0"

# DEAD NAME: MC_DISABLE_HIP_TRANSPORT does not exist in the shipped mooncake
# binary, and a leg launched with it at 1 still installed hip 4x. Kept only
# because it reads as the pair to MC_DISABLE_HIP below, which is the live knob.
# hip is ON here whatever this line says. REQUIRED CHECK: no hipIpcOpenMemHandle failed.
export MC_DISABLE_HIP_TRANSPORT="${MC_DISABLE_HIP_TRANSPORT:-0}"

# MC_DISABLE_HIP is the live knob, left UNSET on purpose: hip on is what this
# shape wants -- selectTransport prefers hip, so KV goes GPU-to-GPU over XGMI
# with no NIC in the path. Set it to 1 for a hip-off A/B, but note it gates
# SELECTION and not install, so no log line confirms it took. See the README.
export MC_DISABLE_HIP="${MC_DISABLE_HIP:-}"

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

# The KV-event ports are PER LEG and on this shape they must differ. Two-node legs
# take engine/leg.sh's defaults (5557/8801) and never meet; here they share one
# network namespace, so identical values make the second leg's bind fail with
# "port_base at N is not available" and it never serves. up.sh forwards these per leg.
export PREFILL_KV_PUB_PORT="${PREFILL_KV_PUB_PORT:-5557}"
export PREFILL_KV_SNAP_PORT="${PREFILL_KV_SNAP_PORT:-8801}"
export DECODE_KV_PUB_PORT="${DECODE_KV_PUB_PORT:-5558}"
export DECODE_KV_SNAP_PORT="${DECODE_KV_SNAP_PORT:-8802}"

# ---------------------------------------------------------------------------
# 5. Features -- see the README before changing these two
# ---------------------------------------------------------------------------

# EVERY FEATURE KNOB THE KIT READS IS PER LEG. The bare MTP/DPA names below are a
# convenience SEED only: engine/up.sh consumes the PREFILL_*/DECODE_* pair, and a
# value set under the bare name alone silently does nothing -- MTP=0 on its own
# still launched the decode leg at mtp=1.

# MTP off HERE ONLY: it is validated on the TWO-NODE symmetric-DPA shape, and this
# wrapper runs dp OFF, so the draft never meets the idle-DP-rank path half the DSA
# patch set exists for -- a different shape, not a smaller one. DECODE_MTP=1 to try
# it, then read the acceptance-length median (2-3 healthy, 4.00 is a repetition loop).
export MTP="${MTP:-0}"
export PREFILL_MTP="${PREFILL_MTP:-$MTP}"
export DECODE_MTP="${DECODE_MTP:-$MTP}"
# DP-ATTENTION IS OFF HERE, AND THAT IS MEASURED, NOT CAUTION. Asymmetric
# (prefill dp1 -> decode dp4) on GLM-5.3-MXFP4 returns a clean FIRST completion
# then garbage on every later one -- repeated `</think>`, digit noise, always
# `length`, at any prompt length. Silent: no faults, no HIP errors, nothing logged.

# The TWO-NODE wrapper instead runs SYMMETRIC dp on both legs, which is clean.
# Symmetric was never tried on one node, so DECODE_DPA=0 is what is validated
# here. For DP-attention on one node try PREFILL_DPA=1 DECODE_DPA=1 first, and
# read the output rather than the exit code.
export DPA="${DPA:-0}"
export PREFILL_DPA="${PREFILL_DPA:-0}"
export DECODE_DPA="${DECODE_DPA:-$DPA}"
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
