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

# --- kvd (KV offload below the GPU cache) -----------------------------------
# KVD=1 runs an infera-kvd daemon next to the prefill engine and points SGLang's
# hierarchical cache at it, so prefixes evicted from the GPU survive in host RAM
# and on local NVMe. KVD=0 is the plain PD deployment and the A/B baseline.
#
# Prefill leg only: SGLang's scheduler issues storage prefetch on the aggregated
# and prefill branches, never on the decode branch, so kvd on a decode leg is
# write-only. infera refuses to wire it there and says so in the log; nothing
# here needs to special-case it.
export KVD="${KVD:-1}"
export KVD_SOCKET="${KVD_SOCKET:-/tmp/infera-kvd/kvd.sock}"
# L3 must be node-local NVMe. Anything shared (NFS, weka) classifies as buffered
# and the reload lands in the TTFT budget instead of under it.
export KVD_L3_DIR="${KVD_L3_DIR:-/your/path/kvd-l3}"
export KVD_RAM_BYTES="${KVD_RAM_BYTES:-64G}"    # kvd L2 arena, pinned host RAM
export KVD_LONG_BYTES="${KVD_LONG_BYTES:-512G}" # kvd L3 budget under KVD_L3_DIR
# O_DIRECT vs buffered for L3. `auto` runs kvd's classifier, which walks the
# mount down to the block device -- and from inside a container that walk ends at
# a /dev/mapper node the container cannot see, so auto falls back to the
# conservative `buffered`. Set `direct` when you know KVD_L3_DIR is local NVMe:
# buffered lets L3 double-book its bytes in the page cache.
export KVD_IO_MODE="${KVD_IO_MODE:-auto}"
# One tablespace slot must hold one whole hicache page, and a page holds every
# layer: GLM-5.2-FP8 at page_size 64 writes 2.74 MiB per KV page and 624 KiB per
# DSA-indexer page. Two pools cover both without a slot's worth of waste on the
# smaller one. A value that outgrows its largest pool is REJECTED, not split --
# smoke.sh checks for that rather than leaving L3 silently empty.
export KVD_TABLESPACE_POOLS="${KVD_TABLESPACE_POOLS:-1M,4M}"
# SGLang's own host tier, in GB PER DP RANK (8 here). It sits between the GPU
# pool and kvd, and is also the staging buffer for L3 reads and write-backs.
# Deliberately smaller than the 54 GB device pool per rank: matching it would pin
# ~870 GB of host RAM for an L2 that kvd's L3 already backs. SGLang warns about
# the ratio; on this base that costs L2 hit rate only -- the prefetch budget is
# 0.5 x host pool, not a function of the host/device difference.
export HICACHE_SIZE="${HICACHE_SIZE:-32}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" 2>/dev/null || true
