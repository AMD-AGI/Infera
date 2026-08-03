#!/usr/bin/env bash
# Shared configuration for GLM-5.2-FP8 on gfx942 with SGLang PD + kv-aware routing.
# Source this from the launch and bench scripts. Every value is overridable; the
# placeholders below are not usable as-is — set at least PREFILL_IP, DECODE_IP,
# MODEL and DATA_DIR for your cluster.
set -uo pipefail

# --- topology ---------------------------------------------------------------
# One prefill node (which also hosts etcd, the router and the bench driver) and
# one decode node. If your nodes resolve by name, setting PREFILL_NODE /
# DECODE_NODE is enough and the IPs are derived from them.
export PREFILL_NODE="${PREFILL_NODE:-node-0}"
export DECODE_NODE="${DECODE_NODE:-node-1}"
export PREFILL_IP="${PREFILL_IP:-$(getent ahostsv4 "$PREFILL_NODE" 2>/dev/null | awk 'NR==1{print $1}')}"
export DECODE_IP="${DECODE_IP:-$(getent ahostsv4 "$DECODE_NODE" 2>/dev/null | awk 'NR==1{print $1}')}"
: "${PREFILL_IP:=127.0.0.1}"
: "${DECODE_IP:=$PREFILL_IP}"

export ETCD_ENDPOINT="${ETCD_ENDPOINT:-${PREFILL_IP}:2379}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"
export PREFILL_PORT="${PREFILL_PORT:-30001}"
export DECODE_PORT="${DECODE_PORT:-31501}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

export ROUTER_URL="${ROUTER_URL:-http://${PREFILL_IP}:${ROUTER_PORT}}"
export PREFILL_URL="${PREFILL_URL:-http://${PREFILL_IP}:${PREFILL_PORT}}"
export DECODE_URL="${DECODE_URL:-http://${DECODE_IP}:${DECODE_PORT}}"

# --- image / container ------------------------------------------------------
# Built by build_image.sh straight from deploy/docker/Dockerfile.sglang.gfx942.
# Do not layer runtime SGLang patches on it.
export IMAGE="${IMAGE:-infera:sglang-gfx942-glm52}"
export CONTAINER="${CONTAINER:-infera-glm52-gfx942}"
export ETCD_CONTAINER="${ETCD_CONTAINER:-infera-glm52-etcd}"

# --- paths ------------------------------------------------------------------
# REPO is bind-mounted at the same path inside the container so these scripts
# resolve identically on the host and in the container.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
export MODEL="${MODEL:-/your/path/GLM-5.2-FP8}"     # local weights dir, mounted read-only
export DATA_DIR="${DATA_DIR:-/your/path/agentic-data}"  # holds the trace dataset, mounted rw

export LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/logs}"
export RESULT_DIR="${RESULT_DIR:-${EXAMPLE_DIR}/results}"

# --- agentic trace ----------------------------------------------------------
# Dataset built by weka_to_agentic_trace.py. OUTPUT_LEN must match the
# --output-len it was built with, or replayed prompt lengths drift each turn.
export TRACE="${TRACE:-${DATA_DIR}/cc_traces_100k.json}"
export OUTPUT_LEN="${OUTPUT_LEN:-220}"

# --- fabric / engine --------------------------------------------------------
export IB_DEVICE="${IB_DEVICE:-mlx5_0}"      # RDMA device Mooncake moves KV over
export MC_GID_INDEX="${MC_GID_INDEX:-3}"     # RoCE GID index on that device
export TP="${TP:-8}"
export DP="${DP:-8}"
export MEM_FRAC="${MEM_FRAC:-0.85}"
export CHUNK="${CHUNK:-131072}"
export MAX_RUNNING="${MAX_RUNNING:-128}"
export MTP="${MTP:-1}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" 2>/dev/null || true
