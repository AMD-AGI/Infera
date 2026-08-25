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
# Usage:  bash cluster/cluster.peermem.sh up | smoke | bench [conc...] | trace_replay | down
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

# Development source overlay. /apps is shared by both nodes, and the engine image's
# WORKDIR is /opt/infera, so mounting this checkout there makes every `python -m infera...`
# invocation import the current working tree. Set INFERA_SRC= to disable the overlay.
# export INFERA_SRC="${INFERA_SRC-$(cd "$KIT/../.." && pwd)}"

# Shared writable trace directory. common.sh bind-mounts this exact absolute path into both
# engine containers, so capture.sh can write there without a tar/docker-cp/SSH fetch stage.
export TRACE_OUT="${TRACE_OUT:-$KIT/profiles}"

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

# ---------------------------------------------------------------------------
# 5. Trace replay — optional, only read by engine/trace_replay.sh
# ---------------------------------------------------------------------------
# AIPerf replays a Mooncake-format production trace at the timestamps it recorded, which is
# the one load in this kit that carries a real shared prefix. Leave AIPERF_TRACE unset and
# nothing here does anything.
#
# The client CANNOT live in the engine container: that image ships Python 3.10 and AIPerf
# requires >= 3.11. It runs from the published NGC image instead — 255 MB, so pulling it is
# not the concern that pulling an engine image is.
export AIPERF_IMAGE="${AIPERF_IMAGE:-nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0}"

# The trace, on a path BOTH this host and $AIPERF_NODE can read. A few MB each, so there is no
# need to clone the repo:
#   B=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces
#   curl -LO $B/conversation_trace.jsonl      # 12031 reqs, avg ISL 12035, avg OSL 343
#   curl -LO $B/toolagent_trace.jsonl         # 23608 reqs, avg ISL  8596, avg OSL 182
# conversation is the default choice; toolagent is closer to an agentic workload. Upstream's
# third file, synthetic_trace.jsonl, has GENERATED (Poisson) arrival times rather than recorded
# ones, which makes it the wrong input for a fixed-schedule replay of real traffic.
# export AIPERF_TRACE="<shared-dir>/conversation_trace.jsonl"

# Which node generates the load. Defaults to the prefill node, which is the simple choice but
# not the neutral one: AIPerf synthesizes and tokenizes every prompt before sending anything,
# and that competes for CPU with the engine's own scheduler and tokenizer processes. Point it
# at any node that can route to $PREFILL_IP:$ROUTER_PORT to remove that interference.
# export AIPERF_NODE="<some-other-host>"

# Artifacts, the generated per-run command file, and the mmap dataset cache. Must be the same
# path on both hosts — trace_replay.sh checks this rather than letting it fail obscurely later.
export AIPERF_OUT="${AIPERF_OUT:-$KIT/aiperf}"

exec bash "$KIT/engine/${1:-up}.sh" "${@:2}"
