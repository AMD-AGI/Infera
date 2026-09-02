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
# Usage:  bash cluster/cluster.dmabuf.sh up | smoke | lm_eval | bench [conc...] |
#                                        trace_replay [prepare|run] | capture | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 0. Before the first bring-up — READ THIS
# ---------------------------------------------------------------------------
# Every value in §2 and §3 is a MEASUREMENT, not a preference, and each one fails
# SILENTLY when it is wrong: a NIC without ODP pins and duplicates the KV pool, and a
# provider mismatch drops everything to TCP at 5-20x the latency. The stack keeps serving
# tokens either way. Run this ON BOTH NODES first and reconcile §2/§3 with what it says:
#
#   IMAGE=<your-infera-image> bash preflight_rdma.sh mode
#     -> "peermem: absent" + "mode B: VIABLE" confirms this wrapper is the right one.
#        If it reports peer-mem present, use cluster/cluster.peermem.sh instead.
#     -> it names the ODP NIC for RDMA_IB_DEVICES and the RoCEv2 index for MC_GID_INDEX,
#        the latter being per-NODE rather than cluster-wide.
#
#   IMAGE=<your-infera-image> DUMP_PATH=<shared-dir> srun -N2 --ntasks-per-node=1 \
#       bash preflight_rdma.sh fabric
#     -> cross-node bandwidth over rdma AND tcp. If the two are close, RDMA is not
#        actually carrying the traffic and everything below measures the fallback path.

# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
# Each node's SSH-reachable name and its DATA-PLANE IP. Do NOT use the
# management/public NIC — the legs advertise these to each other. If ssh to
# compute nodes is blocked, set SSH_CMD (see cluster/README.md).
#
# These are CONTROL-plane addresses and must be routable BETWEEN the nodes; a
# point-to-point /31 link address is not. The KV payload does not ride them — it goes
# over the RDMA device named in §3.
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
# Requires an infera-sglang build NEWER than 0.2.0. The default below is a placeholder.
# The main Infera README names `rocm/infera:sglang-v0.1.1` — that one is too old for this
# kit, so take the newest `sglang-v0.2.x` tag published on Docker Hub instead.
export INFERA_IMAGE="${INFERA_IMAGE:-<infera-sglang-image>}"

# Development source overlay. When both nodes share a filesystem, and because the engine
# image's WORKDIR is /opt/infera, mounting this checkout there makes every
# `python -m infera...` invocation import the current working tree. common.sh verifies the
# mount actually took and refuses to continue if the container imported anything else.
# Set INFERA_SRC= to disable the overlay.
# export INFERA_SRC="${INFERA_SRC-$(cd "$KIT/../.." && pwd)}"

# Shared writable trace directory, required by `capture`. common.sh bind-mounts this exact
# absolute path into both engine containers, so the torch traces land on shared storage
# with no tar/docker-cp/SSH fetch stage. capture.sh checks the mount before it profiles.
export TRACE_OUT="${TRACE_OUT:-$KIT/profiles}"

# MODEL_MOUNT is bind-mounted into the container; MODEL must live under it.
# Shared storage works but loads slowly — ~400 GB over ~280 shards, read by both legs at
# once, so the first bring-up is minutes of silence rather than a hang. That is what
# leg.sh's INFERA_SGLANG_READY_TIMEOUT (3600s) is absorbing; if it still trips, stage the
# weights onto local NVMe per node and repoint this.
export MODEL_MOUNT="<host-dir-holding-the-weights>"
export MODEL="$MODEL_MOUNT/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# If your image injects a host RDMA provider library at entrypoint, set all three: the
# host library (the SYMLINK, so differing per-node builds resolve), the in-container path
# THAT image reads, and the entrypoint. Any one alone silently does nothing.
#
# Why this exists: the image's SGLang layer ships one build of the userspace provider,
# and a host whose kernel module is newer speaks a later ABI. A mismatch does NOT error —
# ibv_get_device_list simply returns ZERO devices, mooncake finds no RDMA and falls back
# to TCP at 5-20x the latency, and the deployment looks healthy the whole time. In this
# mode that also costs you dma-buf itself, which is the only reason to be on wrapper B.
#
# What each of the three does, and how it fails alone:
#   HOST_RDMA_LIB    the host-side SYMLINK, not the resolved .so.1.x.y.z — per-node builds
#                    differ, and the symlink is what makes one line work on both nodes
#   HOST_RDMA_MOUNT  the exact in-container path THAT image's entrypoint reads. Point it
#                    elsewhere and the bind-mount lands where nothing looks: a silent no-op
#   ENTRYPOINT_KEEP  common.sh otherwise starts the container with --entrypoint '', which
#                    skips the injection and leaves the stale provider in place
# The entrypoint is a pass-through when the host path is absent, so leaving this on is safe.
# `docker exec $CTR ibv_devinfo` inside a running container is the check that it worked.
# export HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/lib<provider>.so
# export HOST_RDMA_MOUNT=/host-<provider>/lib<provider>.so ENTRYPOINT_KEEP=1

# ---------------------------------------------------------------------------
# 3. Transport — mode B (dma-buf on the ODP NIC)
# ---------------------------------------------------------------------------
# Copy these straight out of the preflight report's mode-B block — it has
# already checked ODP per card and confirmed dma-buf is compiled into the image.
# ONE device, not a list, and both legs MUST name the same one.
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
# Alongside profiles/, never inside it: that one is the torch-trace mount common.sh binds
# into both engine containers, and mixing the two makes neither directory mean one thing.
export AIPERF_OUT="${AIPERF_OUT:-$KIT/aiperf}"

exec bash "$KIT/engine/${1:-up}.sh" "${@:2}"
