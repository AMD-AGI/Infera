#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  EDIT THIS FILE. Nothing else in this kit needs changing for your site.
# ============================================================================
#
# Usage:  bash engine/up.sh | engine/smoke.sh | engine/bench.sh [conc...] | engine/down.sh
# All of them source this file.

# ---------------------------------------------------------------------------
# 1. Which checkpoint
# ---------------------------------------------------------------------------
# big-mxfp4 | big-fp8 -- the `glm_moe_dsa` models. config.json is field-for-field
# identical to GLM-5.2's except transformers_version, so a released engine already
# serves them through glm4_moe.py and the ordinary deploy/docker/Dockerfile.sglang
# image is all this kit needs.
export VARIANT="${VARIANT:-big-mxfp4}"

# ---------------------------------------------------------------------------
# 2. Site
# ---------------------------------------------------------------------------
# This node's DATA-PLANE IP -- not the management NIC. Clients and the router
# reach the worker here.
export MY_IP="${MY_IP:-<this-node-data-plane-ip>}"

# Weights: MODEL is the directory, MODEL_MOUNT its parent. Give the RESOLVED path
# -- a symlink crossing an NFS mount boundary binds an EMPTY directory into the
# container, surfacing far downstream as "Unrecognized processing class".
# up.sh binds realpath for you.
export MODEL="${MODEL:-<weights-dir>}"

# Engine image. Build it from deploy/docker/Dockerfile.sglang -- see section 1.
export IMAGE="${IMAGE:-<infera-sglang-image>}"

# ---------------------------------------------------------------------------
# 3. Shape
# ---------------------------------------------------------------------------
# TP4 fits ~704 GB of weights and leaves four GPUs free for a second arm on an
# 8-GPU node. TP8 works too; raise GPUS with it.
export TP="${TP:-4}"
export GPUS="${GPUS:-0,1,2,3}"

# Ports. CHECK THESE ARE FREE (`ss -lnt`) -- on a shared node they often are
# not. 2379/2380 in particular are frequently held by somebody else's etcd.
export ETCD_PORT="${ETCD_PORT:-12379}"
export PORT="${PORT:-30000}"
export ROUTER_PORT="${ROUTER_PORT:-8100}"

export CTR="${CTR:-glm53_mix}"
