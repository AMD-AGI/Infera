#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else needs changing.
# ============================================================================
#
# GLM-5.3 (big) 1P1D across TWO nodes -- one prefill, one decode, KV over
# mooncake RDMA. This is the shape the GLM-5.2 kit validated end to end on two
# clusters and both fabric types; only the weights differ here.
#
# It drives the GLM-5.2 kit's engine scripts unchanged -- GLM-5.3 (big) is the
# same architecture, so there is nothing to fork. See the README.
#
# Usage:  bash cluster.2node.sh up | smoke | bench [conc...] | down
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The kit whose engine/ and common.sh actually run. Do not point this at a copy.
export KIT_DIR="${KIT_DIR:-$(cd "$HERE/../sglang_1p1d_glm5.2" && pwd)}"

# ---------------------------------------------------------------------------
# 1. Nodes -- one prefill, one decode
# ---------------------------------------------------------------------------

# Each node's SSH-reachable name and its DATA-PLANE IP -- not the management
# NIC. The legs advertise these to each other for the KV handoff.
export PREFILL_NODE="${PREFILL_NODE:-<prefill-host>}"
export DECODE_NODE="${DECODE_NODE:-<decode-host>}"
export PREFILL_IP="${PREFILL_IP:-<prefill-data-plane-ip>}"
export DECODE_IP="${DECODE_IP:-<decode-data-plane-ip>}"

# ---------------------------------------------------------------------------
# 2. Image and weights (identical on BOTH nodes)
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
export MODEL_MOUNT="${MODEL_MOUNT:-$(dirname "$MODEL")}"   # same path on both nodes
export SERVED="${SERVED:-glm-5.3-mxfp4}"

# ---------------------------------------------------------------------------
# 3. Transport
# ---------------------------------------------------------------------------

# From `preflight_rdma.sh mode`, run on BOTH nodes. Do not guess these.
# MC_GID_INDEX is PER NODE and the two nodes can legitimately differ -- an empty
# slot on one shifts every index after it. A wrong index fails loudly at init on
# every DP rank (`GID is NULL`); the link-local fe80:: GID is never the answer.
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-<ionic_0,ionic_1,...>}"
export MC_GID_INDEX="${MC_GID_INDEX:-<index-from-preflight>}"

# preflight_rdma.sh recommends RDMAV_FORK_SAFE=1 in all three of its modes, and
# engine/leg.sh honours it only when passed in -- so unset here, that recommendation
# is silently dropped. NOT on by default: this shape was validated without it.
# export RDMAV_FORK_SAFE=1

# Leave MC_TE_FILTERS unset in mode A (peer-mem present). It is required only in
# the dma-buf mode, where KV must be pinned to one ODP-capable card.
# export MC_TE_FILTERS="ionic_0"

# ---------------------------------------------------------------------------
# 4. Shape
# ---------------------------------------------------------------------------

# A whole node per leg.
export TP="${TP:-8}"
# Left UNSET on purpose, now that engine/up.sh actually forwards these. A whole
# node per leg means leg.sh's own `seq 0..TP-1` is already right and stays right
# if TP changes; pinning a literal 0..7 here would silently contradict TP=4.
# Set them only to place a leg on a specific subset of a node's cards.
export PREFILL_GPUS="${PREFILL_GPUS:-}"
export DECODE_GPUS="${DECODE_GPUS:-}"

# Separate hosts, so these only need to be free on their own node.
export PREFILL_PORT="${PREFILL_PORT:-30000}"
export DECODE_PORT="${DECODE_PORT:-30001}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
export ROUTER_PORT="${ROUTER_PORT:-8100}"
export ETCD_PORT="${ETCD_PORT:-12379}"

# ---------------------------------------------------------------------------
# 5. Features -- see the README before changing these two
# ---------------------------------------------------------------------------

# EVERY FEATURE KNOB THE KIT READS IS PER LEG. The bare MTP/DPA names below are a
# convenience SEED only: engine/up.sh consumes the PREFILL_*/DECODE_* pair, and a
# value set under the bare name alone silently does nothing and falls back.

# EAGLE MTP, decode leg only -- the GLM-5.2 split, measured here at +36% throughput.
# leg.sh emits MTP(3,1,4) and SPEC_STEPS/SPEC_DRAFT cannot override it: up.sh's
# COMMON_ENV drops them. MTP=0 to A/B. Watch the acceptance-length MEDIAN, 2-3 is
# healthy and 4.00 is the draft predicting a repetition loop. README has the traps.
export MTP="${MTP:-1}"
export PREFILL_MTP="${PREFILL_MTP:-0}"
export DECODE_MTP="${DECODE_MTP:-$MTP}"
# DP-attention must be SYMMETRIC here: dp on both legs. That is a MEASURED
# departure from the GLM-5.2 kit's prefill dp1 -> decode dp8. On GLM-5.3-MXFP4
# asymmetric returns a clean FIRST completion then garbage on every later one --
# repeated `</think>`, digit noise, always `length`. Silent. Re-read the output.
export DPA="${DPA:-1}"
export PREFILL_DPA="${PREFILL_DPA:-$DPA}"
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
