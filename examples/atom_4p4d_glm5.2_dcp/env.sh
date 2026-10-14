#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else in this kit needs changing for your site.
# ============================================================================
#
# Usage:  bash engine/up.sh | engine/smoke.sh | engine/eval.sh | engine/agentx/agentx.sh | engine/down.sh
# All of them source this file. Any variable can also be overridden from the
# environment for one invocation, e.g. `CONC=24 MTP_AL=3.33 bash engine/up.sh`.
# The engine settings are read by up.sh only: smoke.sh, eval.sh and agentx.sh
# reuse the values that run recorded.

# ---------------------------------------------------------------------------
# 1. Site
# ---------------------------------------------------------------------------
# This node's data-plane IP. Router, etcd, engine HTTP and the Mooncake
# handshake bind here; KV itself moves over the ionic RDMA rails.
export NODE_IP="${NODE_IP:-<this-node-ip>}"

# GLM-5.2-MXFP4 weights. The resolved path is bind-mounted and is also the
# served model name.
export MODEL="${MODEL:-<weights-dir>}"

# Infera ATOM image built with the ATOM DCP patches (ATOM_DCP_PATCHES=1); see README "Image".
export IMAGE="${IMAGE:-<infera-atom-image>}"

# Host scratch directory: per-run logs and results (runs/), the InferenceX
# checkout and AgentX / GSM8K caches (cache/). Needs tens of GB.
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# Host libionic provider, mounted so the image entrypoint can match the host
# ionic kernel ABI. Skipped when the file does not exist.
export HOST_LIBIONIC="${HOST_LIBIONIC:-/lib/x86_64-linux-gnu/libionic.so}"

# RoCE GID index of the ionic rails.
export MC_GID_INDEX="${MC_GID_INDEX:-1}"

# GPU memory registration for Mooncake, decided by the node's driver:
#   1 = peer-memory (validated on the reference node with mem util 0.90),
#   0 = HIP DMA-BUF (another node needed this, and passed only at P/D 0.70/0.72).
# Always passed explicitly: the base image bakes in 1.
export MOONCAKE_DISABLE_HIP_DMABUF="${MOONCAKE_DISABLE_HIP_DMABUF:-1}"

# ---------------------------------------------------------------------------
# 2. Shape and load
# ---------------------------------------------------------------------------
# Prefill TP4+DCP4 and decode TP4+DCP4 on disjoint halves of one 8-GPU node.
export PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
export DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"

# Target client concurrency. Decode max-num-seqs and the CUDA-graph sizes of
# both engines are 2 x CONC; restart the engines to change it.
export CONC="${CONC:-48}"

# MTP draft depth: K3 at CONC >= 48, K4 below, as in the reference runs.
export MTP_K="${MTP_K:-$((CONC >= 48 ? 3 : 4))}"

# Forced MTP acceptance length on decode. Empty = real acceptance, the only
# mode whose output is correct; smoke.sh and eval.sh refuse anything else.
# Set it (K3 -> 2.99, K4 -> 3.33) only to reproduce the AgentX reference numbers.
export MTP_AL="${MTP_AL:-}"

export PREFILL_MEM_UTIL="${PREFILL_MEM_UTIL:-0.90}"
export DECODE_MEM_UTIL="${DECODE_MEM_UTIL:-0.90}"

# LMCache CPU KV tier per prefill TP rank, GiB (4 ranks -> 1 TiB pinned host
# memory at the default). 0 turns it off and the prefill uses Mooncake only.
export PREFILL_OFFLOAD_GB="${PREFILL_OFFLOAD_GB:-256}"

# Router --session-affinity on top of round-robin: prefill keeps an agent
# session on one prefill, both also on one decode, off is stateless.
export SESSION_AFFINITY="${SESSION_AFFINITY:-prefill}"

# Extra flags, space-separated, appended last.
export PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"
export DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS:-}"
export ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# 3. Ports and names
# ---------------------------------------------------------------------------
# CHECK THESE ARE FREE (`ss -lnt`). Listeners stay below the ephemeral range
# (32768-60999) and outside Mooncake's random RPC ports (15000-17000), where an
# outgoing socket can grab the port before the listener binds.
export ETCD_PORT="${ETCD_PORT:-23379}"
export ETCD_PEER_PORT="${ETCD_PEER_PORT:-23380}"
export ROUTER_PORT="${ROUTER_PORT:-18000}"
export PREFILL_PORT="${PREFILL_PORT:-19001}"
export DECODE_PORT="${DECODE_PORT:-19002}"
# Infera readiness probe; both workers share the host network, so they differ.
export PREFILL_READINESS_PORT="${PREFILL_READINESS_PORT:-19011}"
export DECODE_READINESS_PORT="${DECODE_READINESS_PORT:-19012}"
# Mooncake handshake: base + TP rank, so 21301-21304 and 21311-21314.
export PREFILL_HANDSHAKE_PORT="${PREFILL_HANDSHAKE_PORT:-21301}"
export DECODE_HANDSHAKE_PORT="${DECODE_HANDSHAKE_PORT:-21311}"

export ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
# Container name prefix: <CTR>-etcd, -prefill, -decode, -router, -agentx, -eval.
export CTR="${CTR:-glm52_4p4d_dcp}"

# ---------------------------------------------------------------------------
# 4. Timeouts, seconds
# ---------------------------------------------------------------------------
# Cold start loads weights, JITs AITER kernels and captures graphs.
export READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# The driver can hold a removed engine's VRAM for 15-40 minutes.
export VRAM_RELEASE_TIMEOUT="${VRAM_RELEASE_TIMEOUT:-3600}"
