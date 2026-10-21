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

# Infera ATOM image built on the ATOM nightly base (ATOM_NIGHTLY=1); see README "Image".
export IMAGE="${IMAGE:-<infera-atom-image>}"

# Host scratch directory: per-run logs and results (runs/), the InferenceX
# checkout and AgentX / GSM8K caches (cache/). Needs tens of GB.
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# Host libionic provider, mounted so the image entrypoint can match the host
# ionic kernel ABI. Skipped when the file does not exist on the engine's node.
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
# Prefill TP1 x PP4 and decode TP4+DCP4 on disjoint halves of one 8-GPU node.
export PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
export DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"

# Target client concurrency. Decode max-num-seqs and the CUDA-graph sizes are
# 2 x CONC (also of a DCP prefill); restart the engines to change it.
export CONC="${CONC:-48}"

# MTP draft depth: K3 at CONC >= 48, K4 below, as in the reference runs.
export MTP_K="${MTP_K:-$((CONC >= 48 ? 3 : 4))}"

# Forced MTP acceptance length on decode. Empty = real acceptance, the only
# mode whose output is correct; smoke.sh and eval.sh refuse anything else.
# Set it (K3 -> 2.99, K4 -> 3.33) only to reproduce the AgentX reference numbers.
export MTP_AL="${MTP_AL:-}"

# Memory, chunk budgets, block size, layer split and offload load length below
# follow ATOM's GLM-5.2 AgentX recipe (/app/ATOM/recipes/Agentic-GLM-5.2.md in
# the image) and InferenceX PR #4045. The decode's 0.94 and 2048, with the
# shared communicator groups worker.sh sets, give it ~22% more KV blocks, which
# keeps the agents' prefixes on the decode up to about C105 (README "The recipe").
export PREFILL_MEM_UTIL="${PREFILL_MEM_UTIL:-0.85}"
export DECODE_MEM_UTIL="${DECODE_MEM_UTIL:-0.94}"

# Chunk budgets (--max-num-batched-tokens). ATOM's compile cache
# (/root/.cache/atom) does not key on them, so worker.sh keeps it in a volume per
# value (README gotcha 11). A full decode step takes (MTP_K + 1) x 2 x CONC
# tokens, 896 at CONC 112; up.sh refuses a decode budget below that.
export PREFILL_MAX_BATCHED_TOKENS="${PREFILL_MAX_BATCHED_TOKENS:-8192}"
export DECODE_MAX_BATCHED_TOKENS="${DECODE_MAX_BATCHED_TOKENS:-2048}"

# KV block size of both legs, which must match: the PP prefill lays the tokens
# each decode DCP rank owns into that rank's pages. Block x DCP4 must divide
# the LMCache chunk of 256 tokens.
export BLOCK_SIZE="${BLOCK_SIZE:-16}"

# Prefill parallelism: 4 = TP1 x PP4, eager, with PREFILL_PP_PARTITION layers
# per stage. 1 = TP4 + DCP4 with CUDA graphs, slower under AgentX load (README
# "Reference numbers"); its validated settings were PREFILL_MEM_UTIL=0.90,
# PREFILL_MAX_BATCHED_TOKENS=16384, BLOCK_SIZE=64, OFFLOAD_MIN_LOAD_TOKENS=8192
# and PREFILL_SEND_WORKERS empty.
export PREFILL_PP="${PREFILL_PP:-4}"
export PREFILL_PP_PARTITION="${PREFILL_PP_PARTITION:-20,20,20,18}"

# Mooncake send workers per prefill rank (num_worker_threads). Empty = ATOM's
# default, 16, run with one QP per endpoint (MC_NUM_QP_PER_EP=1, worker.sh);
# with Mooncake's default two QPs 16 workers hung an RDMA write of the PP
# prefill (README gotcha 13).
export PREFILL_SEND_WORKERS="${PREFILL_SEND_WORKERS-}"

# CPU KV tier behind the prefill:
#   store    Mooncake Store, as InferenceX PR #4045: one master and two owners
#            of STORE_OWNER_GB each, one per NUMA node, one pool shared by the
#            four PP stages, read and written over RDMA (engine/store.sh,
#            README gotcha 14)
#   lmcache  LMCache, PREFILL_OFFLOAD_GB per PP stage, as GTT (gotcha 4)
#   off      Mooncake P->D only
export PREFILL_CPU_TIER="${PREFILL_CPU_TIER:-store}"
export STORE_OWNER_GB="${STORE_OWNER_GB:-768}"
# The prefill's rails in PREFILL_GPUS order: the owners register on all of
# them, and stage i's Store client uses the i-th, so each stage reaches the
# owners on its own rail (the fabric routes same-numbered rails only, gotcha
# 10). Explicit, because ATOM finds a GPU's NIC from the PCI tree, which a VM
# can flatten.
export STORE_OWNER_DEVICES="${STORE_OWNER_DEVICES:-ionic_0,ionic_1,ionic_2,ionic_3}"
export PREFILL_OFFLOAD_GB="${PREFILL_OFFLOAD_GB:-256}"
# Shortest CPU-tier hit the prefill loads; shorter hits are recomputed.
export OFFLOAD_MIN_LOAD_TOKENS="${OFFLOAD_MIN_LOAD_TOKENS:-0}"

# Router --session-affinity on top of round-robin: prefill keeps an agent
# session on one prefill, both also on one decode, off is stateless.
export SESSION_AFFINITY="${SESSION_AFFINITY:-prefill}"

# Router data plane: rust (the image's infera-router binary) or python. Both
# route ATOM PD the same way.
export ROUTER_BACKEND="${ROUTER_BACKEND:-rust}"

# Extra flags, space-separated, appended last.
export PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"
export DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS:-}"
export ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:-}"
# Extra engine container environment, space-separated KEY=VALUE pairs, appended
# last; e.g. ATOM_METRICS_PER_REQUEST_CONTEXT=1 for ATOM's per-request context
# gauges, which the image leaves off (README "Image").
export PREFILL_EXTRA_ENV="${PREFILL_EXTRA_ENV:-}"
export DECODE_EXTRA_ENV="${DECODE_EXTRA_ENV:-}"

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
# Mooncake handshake: base + PP stage or TP rank, so 21301-21304 and 21311-21314.
export PREFILL_HANDSHAKE_PORT="${PREFILL_HANDSHAKE_PORT:-21301}"
export DECODE_HANDSHAKE_PORT="${DECODE_HANDSHAKE_PORT:-21311}"
# Mooncake Store: master RPC, metadata HTTP, metrics, owner i on base + i.
export STORE_MASTER_PORT="${STORE_MASTER_PORT:-26051}"
export STORE_METADATA_PORT="${STORE_METADATA_PORT:-26080}"
export STORE_METRICS_PORT="${STORE_METRICS_PORT:-26090}"
export STORE_OWNER_PORT="${STORE_OWNER_PORT:-26052}"

export ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
# Container name prefix: <CTR>-etcd, -prefill, -decode, -router, -agentx, -eval,
# -store-master, -store-owner0, -store-owner1.
export CTR="${CTR:-glm52_4p4d}"

# ---------------------------------------------------------------------------
# 4. Timeouts, seconds
# ---------------------------------------------------------------------------
# Cold start loads weights, JITs AITER kernels and captures graphs.
export READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# The driver can hold a removed engine's VRAM for 15-40 minutes.
export VRAM_RELEASE_TIMEOUT="${VRAM_RELEASE_TIMEOUT:-3600}"
