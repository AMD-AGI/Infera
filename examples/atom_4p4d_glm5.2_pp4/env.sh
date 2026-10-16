#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. 4P4D with a TP1 x PP4 prefill, built on the 4P4D kit.
# ============================================================================
#
# Usage: bash engine/up.sh | engine/smoke.sh | engine/eval.sh | engine/agentx.sh
#   | engine/down.sh
# All of them source this file, which then sources ../atom_4p4d_glm5.2_dcp/env.sh:
# everything not set here (decode, LMCache, ports, timeouts) is the 4P4D kit's.
# Any variable can be overridden from the environment for one invocation.

# ---------------------------------------------------------------------------
# 1. Site
# ---------------------------------------------------------------------------
export NODE_IP="${NODE_IP:-<this-node-ip>}"
export MODEL="${MODEL:-<weights-dir>}"
export IMAGE="${IMAGE:-<infera-atom-image>}"
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# ---------------------------------------------------------------------------
# 2. Prefill recipe, from ATOM's GLM-5.2 AgentX recipe
#    (/app/ATOM/recipes/Agentic-GLM-5.2.md in the image)
# ---------------------------------------------------------------------------
export PREFILL_PP=4
export PREFILL_PP_PARTITION="${PREFILL_PP_PARTITION:-20,20,20,18}"
export PREFILL_MEM_UTIL="${PREFILL_MEM_UTIL:-0.85}"
export PREFILL_MAX_BATCHED_TOKENS="${PREFILL_MAX_BATCHED_TOKENS:-8192}"
export OFFLOAD_MIN_LOAD_TOKENS="${OFFLOAD_MIN_LOAD_TOKENS:-0}"
# Both legs: the non-DCP prefill lays the tokens each decode DCP rank owns into
# that rank's pages, which assumes one block size on both sides.
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
# ATOM's default of 16 hangs an RDMA write within the first requests; one
# misses the 120 s transfer deadline on 300k-token prompts (README).
export PREFILL_SEND_WORKERS="${PREFILL_SEND_WORKERS:-4}"

export CTR="${CTR:-glm52_4p4d_pp4}"

export BASE_KIT
BASE_KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../atom_4p4d_glm5.2_dcp" && pwd)"
# shellcheck source=../atom_4p4d_glm5.2_dcp/env.sh
source "$BASE_KIT/env.sh"
