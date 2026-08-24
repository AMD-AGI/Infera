#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Site wrapper for the crsuse2-m2m (spur) cluster — 2-node 1P1D for validating
# upstream sglang PR #33970 (mooncake KV transfer waits on the prefill forward).
#
# Shaped from cluster/cluster.dmabuf.sh. Mode B, on mlx5_0, exactly as
# `python -m infera.tools.preflight.mooncake_mode` reports on these nodes:
#
#     [B] ibv_reg_dmabuf_mr on the ODP NIC (no-pin) -- viable * best
#         NICs: mlx5_0    MC_GID_INDEX=3    MOONCAKE_DISABLE_HIP_DMABUF=0
#     [A] BLOCKED: no peer-mem module loaded
#     [C] viable but needs a computed KV cap (ionic rails have no ODP)
#
# Mode B's bandwidth regression (KV on 1x mellanox@200G instead of 8x ionic@400G)
# is ACCEPTED here, with the user's agreement: #33970's claim is a correctness
# race, not throughput. If anything a slower link widens the race window, so it
# is the conservative choice for reproducing the defect. Any throughput number
# taken on this rig must be labelled as a 200 Gb/s-link figure.
#
# Independently corroborated before use: a mooncake TransferEngine MVP across
# these two nodes transfers 8 MiB byte-correct over mlx5_0 (rc=0), and mooncake
# auto-selects GID index 3. Over ionic_0 the same MVP silently falls back to TCP
# ("Found 0 HCAs") because the container's libibverbs rejects the host's ionic
# driver ABI -- see rounds/r03-newcluster/mooncake_mvp.md.
#
# Usage:  bash cluster.spur.sh up | smoke | bench [conc...] | down
#         ARM=stock|patched  selects the sglang tree in the containers (see up_ab.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT="${KIT:-/home/yihou/dev/git/infera.upstream.pr.verify/examples/sglang_1p1d_glm5.2}"

# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
# ssh to compute nodes is blocked here (sshd AllowUsers whitelist), so reach them
# through spur. up.sh supports exactly this via SSH_CMD -- see its line 18.
export PREFILL_NODE="crsuse2-m2m-237"
export DECODE_NODE="crsuse2-m2m-106"

# Data-plane IPs. On this cluster the ONLY routed interface is ens3, and mlx5_0
# is bound to it -- the ionic rails are IPv6-only and unusable from the container
# (see above). So the data plane and the management network are the same NIC here.
export PREFILL_IP="10.245.154.191"
export DECODE_IP="10.245.159.121"

export SSH_CMD="$HERE/spur_ssh.sh"
export NODE_MAP="${NODE_MAP:-crsuse2-m2m-237=58799,crsuse2-m2m-106=58800}"

# $HOME is NFS and identical on both nodes, so one path satisfies "same path on
# both nodes".
export KIT_DIR="$KIT"

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
export INFERA_IMAGE="${INFERA_IMAGE:-infera-local:sglang-prverify-20260824}"

# Weights are node-local on /mnt/m2m_nobackup (28T), present on BOTH nodes.
#
# FP8, not the mxfp4 copy sitting next to it. `mlx-community__GLM-5.2-mxfp4` is an
# MLX-format conversion: its quantization_config is {"mode": "mxfp4"} with no
# `quant_method` key, so sglang reads an empty string and dies with
#   ValueError: Unknown quantization method: . Must be one of [... 'mxfp4']
# -- confusingly listing mxfp4 as valid. It is the FORMAT that is wrong, not the
# quantization. zai-org__GLM-5.2-FP8 has a proper `quant_method: fp8` and the same
# GlmMoeDsaForCausalLM architecture. 708 GB at TP8 is ~89 GB/card on 288 GB MI355X.
export MODEL_MOUNT="${MODEL_MOUNT:-/mnt/m2m_nobackup/models}"
export MODEL="${MODEL:-$MODEL_MOUNT/zai-org__GLM-5.2-FP8}"
export TOKENIZER="$MODEL"
export SERVED="${SERVED:-glm5.2-fp8}"

# ---------------------------------------------------------------------------
# 3. Transport — mode B (dma-buf on the ODP NIC), straight from preflight
# ---------------------------------------------------------------------------
export RDMA_IB_DEVICES="mlx5_0"
export MC_MS_AUTO_DISC=0
export MC_MS_FILTERS="$RDMA_IB_DEVICES"
export MC_GID_INDEX=3              # RoCE v2 IPv4 on mlx5_0; NOT n06-33's 1
export MOONCAKE_DISABLE_HIP_DMABUF=0
export RDMAV_FORK_SAFE=1           # trap 2: mooncake fails without it

# ---------------------------------------------------------------------------
# 4. Deployment shape
# ---------------------------------------------------------------------------
# TP8 per leg: one whole node each, which is the shape the PR targets. GLM-5.2
# FP8 is 708 GB, so TP8 on 288 GB MI355X is ~89 GB/card.
export TP="${TP:-8}"
export CTX="${CTX:-262144}"

# CHUNK is the GLOBAL per-step prefill budget. With DP-attention on, sglang
# DIVIDES it by dp_size -- trap 7, which invalidated the previous session's
# needle arithmetic. The probe must read the RESOLVED value off the leg's own
# server_args line, never this number.
export CHUNK="${CHUNK:-131072}"

export PREFILL_DPA="${PREFILL_DPA:-0}"   # pure TP on prefill
export DECODE_DPA="${DECODE_DPA:-1}"
export PREFILL_MTP=0
export DECODE_MTP="${DECODE_MTP:-1}"

# kv-aware routing and kvd are orthogonal to the KV-transfer race, and each adds
# moving parts and host-port surface. Off, to keep the A/B to one variable.
#
# NOTE round-robin alone does not start: common.sh:91 passes
# --router-tokenizer-path only on the kv-aware branch, but infera/server/args.py:140
# declares it required=True unconditionally, so the router dies with
#   error: the following arguments are required: --router-tokenizer-path
# That is a bug in the kit, not in this experiment. common.sh has no hook to inject
# an extra flag, and patching the shared kit is out of scope here, so `up` will fail
# at the router step -- both LEGS come up fine. Start the router afterwards with
# scripts/router.sh, which passes the flag. Switching to kv-aware would also work but
# adds the routing variable this A/B deliberately excludes. Reported, not fixed.
export ROUTER_POLICY="${ROUTER_POLICY:-round-robin}"
export KVAWARE="${KVAWARE:-0}"
export PREFILL_KVD="${PREFILL_KVD:-0}"
export DECODE_KVD=0

export GMU_PREFILL="${GMU_PREFILL:-0.70}"
export GMU_DECODE="${GMU_DECODE:-0.85}"

exec bash "$KIT/engine/${1:-up}.sh" "${@:2}"
