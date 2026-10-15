#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. 8P4D on two nodes, built on the 4P4D kit.
# ============================================================================
#
# Usage, on the head node: bash engine/up.sh | engine/smoke.sh | engine/eval.sh
#   | engine/agentx.sh | engine/down.sh
# All of them source this file, which then sources ../atom_4p4d_glm5.2_dcp/env.sh:
# everything not set here (recipe knobs, ports, timeouts) is the 4P4D kit's.
# Any variable can be overridden from the environment for one invocation, e.g.
# `CONC=84 MTP_AL=2.99 bash engine/up.sh`.

# ---------------------------------------------------------------------------
# 1. Site. Both kits, MODEL and WORK_DIR must have the same paths on both nodes.
# ---------------------------------------------------------------------------
# Head node data-plane IP: etcd, router, prefill0, decode and the clients.
export NODE_IP="${NODE_IP:-<head-node-ip>}"

# Second node, prefill1 only. PREFILL1_HOST is its ssh destination
# ([user@]host, key-based login, no port): the head node drives its docker
# through DOCKER_HOST=ssh://$PREFILL1_HOST. PREFILL1_IP is its data-plane IP.
export PREFILL1_HOST="${PREFILL1_HOST:-<second-node-ssh>}"
export PREFILL1_IP="${PREFILL1_IP:-<second-node-ip>}"

export MODEL="${MODEL:-<weights-dir>}"
# Same image ID on both nodes: build once, then docker save | docker load.
export IMAGE="${IMAGE:-<infera-atom-image>}"
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# ---------------------------------------------------------------------------
# 2. Shape and load
# ---------------------------------------------------------------------------
# Head node: prefill0 and decode split its eight GPUs, as in 4P4D.
export PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
export DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"
# Second node: prefill1. Its other four GPUs stay idle.
export PREFILL1_GPUS="${PREFILL1_GPUS:-0,1,2,3}"

# Target client concurrency (reference points 84 and 96).
export CONC="${CONC:-96}"
# PREFILL_OFFLOAD_GB keeps the 4P4D default, 256 GiB per rank: 1 TiB of GTT on
# each node, the most one node fits (README gotchas).

export CTR="${CTR:-glm52_8p4d_dcp}"

# ---------------------------------------------------------------------------
# 3. 4P4D kit hooks. Fixed by this layout; not site values.
# ---------------------------------------------------------------------------
export PREFILL_NUM_WORKERS=2 IS_MULTINODE=true
export REMOTE_PREFILLS
REMOTE_PREFILLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/engine/prefill1.sh"
export BASE_KIT
BASE_KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../atom_4p4d_glm5.2_dcp" && pwd)"
# shellcheck source=../atom_4p4d_glm5.2_dcp/env.sh
source "$BASE_KIT/env.sh"
