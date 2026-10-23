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
# The scripts run on the prefill node; the decode node is driven over ssh.

# ---------------------------------------------------------------------------
# 1. Site
# ---------------------------------------------------------------------------
# Prefill node (this one): two prefill workers, etcd, router and the clients.
export PREFILL_NODE_IP="${PREFILL_NODE_IP:-<prefill-node-ip>}"
# Decode node: one decode worker. DECODE_SSH is how this node reaches it
# (passwordless ssh, docker without sudo); this kit must sit at the same path
# there.
export DECODE_NODE_IP="${DECODE_NODE_IP:-<decode-node-ip>}"
export DECODE_SSH="${DECODE_SSH:-$DECODE_NODE_IP}"

# GLM-5.2-MXFP4 weights, at the same path on both nodes (bind-mounted
# read-only, also the router's tokenizer).
export MODEL="${MODEL:-<weights-dir>}"
export SERVED_MODEL="${SERVED_MODEL:-glm5.2-mxfp4}"

# Infera SGLang image built from deploy/docker/Dockerfile.sglang, present on
# both nodes under this tag (README "Image").
export IMAGE="${IMAGE:-<infera-sglang-image>}"

# Prefill-node scratch directory: per-run logs and results (runs/), the
# InferenceX checkout and the AgentX / GSM8K client caches (cache/).
export WORK_DIR="${WORK_DIR:-<scratch-dir>}"

# KV transfer rails (Mooncake RDMA) and their routable RoCEv2 GID index.
# MOONCAKE_DISABLE_HIP_DMABUF=0 registers VRAM through dma-buf; set 1 on a
# fabric whose provider cannot (README note 5).
export RDMA_DEVICES="${RDMA_DEVICES:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"
export MOONCAKE_DISABLE_HIP_DMABUF="${MOONCAKE_DISABLE_HIP_DMABUF:-0}"

# ---------------------------------------------------------------------------
# 2. Engines
# ---------------------------------------------------------------------------
# Two TP1 x PP4 prefills (4 GPUs each) on the prefill node, one TP + DP-attention
# decode on the decode node. PREFILL_WORKERS=1 starts prefill 0 only; a 4-GPU
# DECODE_GPUS makes a TP4 + DP4-attention decode. Which shape serves which
# concurrency best: README "Reference numbers".
export PREFILL_WORKERS="${PREFILL_WORKERS:-2}"
export PREFILL0_GPUS="${PREFILL0_GPUS:-0,1,2,3}"
export PREFILL1_GPUS="${PREFILL1_GPUS:-4,5,6,7}"
export DECODE_GPUS="${DECODE_GPUS:-0,1,2,3,4,5,6,7}"

# Forced MTP acceptance length on the decode. Empty = real acceptance, the only
# mode whose output is correct: use it for smoke.sh and eval.sh. 3.61
# reproduces the AgentX reference numbers (InferenceX's GLM-5.2 value).
export SIM_ACCEPT_LEN="${SIM_ACCEPT_LEN:-}"

# Tokens one held request is worth when the router balances the decode ranks
# (--decode-request-weight); 0 balances by input tokens only (README "Decode
# request balancing").
export DECODE_REQUEST_WEIGHT="${DECODE_REQUEST_WEIGHT:-1000000}"

# Extra flags, space-separated, appended last.
export PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"
export DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS:-}"
export ROUTER_EXTRA_ARGS="${ROUTER_EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# 3. Benchmarks
# ---------------------------------------------------------------------------
# AgentX client concurrency (agentx.sh only; the engines do not depend on it).
export CONC="${CONC:-176}"
# GSM8K request concurrency (eval.sh; InferenceX's default).
export EVAL_CONC="${EVAL_CONC:-64}"

# ---------------------------------------------------------------------------
# 4. Ports and names
# ---------------------------------------------------------------------------
# CHECK THESE ARE FREE (`ss -lnt`) on both nodes. Prefill i uses
# PREFILL_PORT + 256 i, BOOTSTRAP_PORT + i, KV_SNAPSHOT_PORT + i, and publishes
# KV events of its DP rank r on KV_EVENTS_PORT + 8 i + r. SGLang also takes
# the 11 ports from each engine port + 233 for its own processes.
export ETCD_PORT="${ETCD_PORT:-24701}"
export ETCD_PEER_PORT="${ETCD_PEER_PORT:-24702}"
export ROUTER_PORT="${ROUTER_PORT:-28300}"
export PREFILL_PORT="${PREFILL_PORT:-27201}"
export DECODE_PORT="${DECODE_PORT:-27301}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-28398}"
export KV_EVENTS_PORT="${KV_EVENTS_PORT:-27757}"
export KV_SNAPSHOT_PORT="${KV_SNAPSHOT_PORT:-29051}"

export ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
# Container name prefix: <CTR>-etcd, -prefill0, -prefill1, -router, -agentx,
# -eval on the prefill node, <CTR>-decode on the decode node.
export CTR="${CTR:-glm52_2p1d}"
