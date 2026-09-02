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
# flash-mxfp4 | flash-fp8 | big-mxfp4 | big-fp8
#
# THE VARIANT DECIDES WHICH IMAGE YOU NEED, because GLM-5.3 is two unrelated
# architectures sharing a product name:
#
#   flash-*  model_type glm5_next, GlmMoeDsaForCausalLM's opposite number --
#            hybrid KDA-linear + DSA attention, mHC, natively multimodal.
#            Exists in NO released sglang. Needs the image built from
#            deploy/docker/Dockerfile.sglang.glm53, which overlays sglang
#            PR #36607 at a pinned SHA.
#   big-*    model_type glm_moe_dsa. Field-for-field identical to GLM-5.2
#            except transformers_version, so the released engine already
#            serves it. Needs the ordinary deploy/docker/Dockerfile.sglang
#            image.
#
# Crossing them fails at CONFIG LOAD, not at inference:
#   ValueError: The checkpoint you are trying to load has model type
#   `glm5_next` but Transformers does not recognize this architecture.
# That message names transformers and invites the wrong fix. The missing
# component is sglang, not transformers.
export VARIANT="${VARIANT:-flash-mxfp4}"

# ---------------------------------------------------------------------------
# 2. Site
# ---------------------------------------------------------------------------
# This node's DATA-PLANE IP -- not the management NIC. Clients and the router
# reach the worker here.
export MY_IP="${MY_IP:-<this-node-data-plane-ip>}"

# Weights. MODEL must be the directory, MODEL_MOUNT its parent.
#
# Resolve symlinks yourself. On the reference cluster /apps/data/models is a
# symlink to /perf_apps/data/models and /perf_apps is a SEPARATE NFS mount, so
# bind-mounting the symlink's parent gives the container an empty directory and
# every path under it dangles. The failure surfaces far downstream as
# "Unrecognized processing class", because config.json is the one file docker
# still resolves. up.sh binds `realpath` output for this reason.
export MODEL="${MODEL:-/apps/data/models/GLM-5.3-Flash-MXFP4}"

# Engine image. Must match VARIANT -- see section 1.
export IMAGE="${IMAGE:-<infera-sglang-image>}"

# ---------------------------------------------------------------------------
# 3. Shape
# ---------------------------------------------------------------------------
# TP4 is what AMD validated for the Flash MXFP4 checkpoint, and it leaves four
# GPUs free for a second arm on an 8-GPU node. TP8 works too; raise GPUS with it.
export TP="${TP:-4}"
export GPUS="${GPUS:-0,1,2,3}"

# Ports. CHECK THESE ARE FREE (`ss -lnt`) -- on a shared node they often are
# not. 2379/2380 in particular are frequently held by somebody else's etcd.
export ETCD_PORT="${ETCD_PORT:-12379}"
export PORT="${PORT:-30000}"
export ROUTER_PORT="${ROUTER_PORT:-8100}"

export CTR="${CTR:-glm53_mix}"
