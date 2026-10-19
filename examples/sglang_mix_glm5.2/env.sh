#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else in this kit needs changing for your site.
# ============================================================================
#
# Sourced by every script in engine/. Any variable can be overridden from the
# environment for one invocation, e.g. `SIM_ACCEPT_LEN=3.61 bash engine/up.sh`.

# ---------------------------------------------------------------------------
# 1. Site
# ---------------------------------------------------------------------------
# This node's data-plane IP; router, etcd and the engine bind here.
export NODE_IP="${NODE_IP:-<this-node-ip>}"

# GLM-5.2-MXFP4 weights (bind-mounted read-only, also the router's tokenizer).
export MODEL="${MODEL:-<weights-dir>}"
export SERVED_MODEL="${SERVED_MODEL:-glm5.2-mxfp4}"

# Infera SGLang image built from deploy/docker/Dockerfile.sglang (README "Image").
export IMAGE="${IMAGE:-<infera-sglang-image>}"

# Host scratch directory: per-run logs and results (runs/), the InferenceX
# checkout and the AgentX / GSM8K client caches (cache/). Needs tens of GB.
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# ---------------------------------------------------------------------------
# 2. Engine
# ---------------------------------------------------------------------------
# One TP4 + DP4-attention worker on four GPUs.
export GPUS="${GPUS:-0,1,2,3}"

# Forced MTP acceptance length. Empty = real acceptance, the only mode whose
# output is correct: use it for smoke.sh and eval.sh. 3.61 reproduces the AgentX
# reference numbers (InferenceX's GLM-5.2 value).
export SIM_ACCEPT_LEN="${SIM_ACCEPT_LEN:-}"

# Extra flags, space-separated, appended last.
export ENGINE_EXTRA_ARGS="${ENGINE_EXTRA_ARGS:-}"
export ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# 3. Benchmark
# ---------------------------------------------------------------------------
# AgentX client concurrency (agentx.sh only; the engine does not depend on it).
export CONC="${CONC:-48}"

# ---------------------------------------------------------------------------
# 4. Ports and names
# ---------------------------------------------------------------------------
# CHECK THESE ARE FREE (`ss -lnt`). DP rank r publishes KV events on
# KV_EVENTS_PORT + r.
export ETCD_PORT="${ETCD_PORT:-24601}"
export ETCD_PEER_PORT="${ETCD_PEER_PORT:-24602}"
export ROUTER_PORT="${ROUTER_PORT:-28200}"
export ENGINE_PORT="${ENGINE_PORT:-27101}"
export KV_EVENTS_PORT="${KV_EVENTS_PORT:-27657}"
export KV_SNAPSHOT_PORT="${KV_SNAPSHOT_PORT:-28951}"

export ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
# Container name prefix: <CTR>-etcd, -engine, -router, -agentx, -eval.
export CTR="${CTR:-glm52_mix}"
