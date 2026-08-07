#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Shared configuration for GLM-5.2-FP8 on two gfx942 nodes: SGLang prefill/decode
# disaggregation over Mooncake RDMA with Infera kv-aware routing. Sourced by every
# script here, so you should not need to edit any of them. Set at least
# PREFILL_IP, DECODE_IP and MODEL for your cluster.
#
# The engine values below are the tuned recipe (README "The tuned recipe"), not
# SGLang defaults; each carries the measurement that chose it.
set -uo pipefail

# --- topology ---------------------------------------------------------------
# One prefill node, which also hosts etcd and the router, and one decode node.
# If your nodes resolve by name, setting PREFILL_NODE / DECODE_NODE is enough.
export PREFILL_NODE="${PREFILL_NODE:-node-0}"
export DECODE_NODE="${DECODE_NODE:-node-1}"
export PREFILL_IP="${PREFILL_IP:-$(getent ahostsv4 "$PREFILL_NODE" 2>/dev/null | awk 'NR==1{print $1}')}"
export DECODE_IP="${DECODE_IP:-$(getent ahostsv4 "$DECODE_NODE" 2>/dev/null | awk 'NR==1{print $1}')}"

export ETCD_ENDPOINT="${ETCD_ENDPOINT:-${PREFILL_IP}:2379}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"
export PREFILL_PORT="${PREFILL_PORT:-30001}"
export DECODE_PORT="${DECODE_PORT:-31501}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

export ROUTER_URL="${ROUTER_URL:-http://${PREFILL_IP}:${ROUTER_PORT}}"
export PREFILL_URL="${PREFILL_URL:-http://${PREFILL_IP}:${PREFILL_PORT}}"
export DECODE_URL="${DECODE_URL:-http://${DECODE_IP}:${DECODE_PORT}}"

# Called by every script that dials one of the addresses above, and by no other --
# build_image.sh and host_container.sh source this file and need no IP at all.
#
# The tempting default, loopback for a node that does not resolve, is the worst
# kind of wrong here: etcd runs on the prefill node, so THAT leg registers happily
# and its whole node looks healthy while the decode leg finds nothing listening on
# its own loopback -- 20 minutes later, since registration comes after the weights
# load. A half-set pair is worse still: DECODE_IP falling back to PREFILL_IP has
# the decode leg advertise the prefill node's address, and then BOTH legs register
# and only a real request finds the hole.
require_ips() {
  local bad=0
  [[ -n "$PREFILL_IP" ]] || { echo "[env] PREFILL_IP unset and '$PREFILL_NODE' does not resolve" >&2; bad=1; }
  [[ -n "$DECODE_IP" ]]  || { echo "[env] DECODE_IP unset and '$DECODE_NODE' does not resolve" >&2; bad=1; }
  (( bad == 0 )) || {
    echo "[env] export both, or point PREFILL_NODE/DECODE_NODE at names that resolve" >&2
    exit 1
  }
}

# --- image / container ------------------------------------------------------
# Built by build_image.sh straight from deploy/docker/Dockerfile.sglang.gfx942.
# Do not layer runtime SGLang patches on it.
export IMAGE="${IMAGE:-infera:sglang-gfx942-glm52}"
export CONTAINER="${CONTAINER:-infera-glm52-gfx942}"
export ETCD_CONTAINER="${ETCD_CONTAINER:-infera-glm52-etcd}"

# --- paths ------------------------------------------------------------------
# REPO is bind-mounted at the same path inside the container, so these scripts
# resolve identically on the host and in the container.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
export MODEL="${MODEL:-/your/path/GLM-5.2-FP8}"   # local weights dir, mounted read-only
export LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/logs}"
export RESULT_DIR="${RESULT_DIR:-${EXAMPLE_DIR}/results}"

# --- fabric -----------------------------------------------------------------
# One rail carries the KV transfer. Striping it over every NIC was measured 11.9%
# slower and cannot help: KV uses 4.5% of a single 200 Gb/s port on this workload.
export IB_DEVICE="${IB_DEVICE:-mlx5_0}"
export MC_GID_INDEX="${MC_GID_INDEX:-3}"          # RoCE GID index on that device
export TP="${TP:-8}"
export DP="${DP:-8}"

# --- engine -----------------------------------------------------------------
export MEM_FRAC="${MEM_FRAC:-0.85}"
export MAX_RUNNING="${MAX_RUNNING:-128}"

# Aggregate value, NOT per rank: dp-attention splits it CHUNK/DP, so 8192 is
# 1,024/rank. Largest single lever on this deployment -- 1,024/rank beat the
# 16,384/rank this recipe used before by 23.8% on duration and 34.4% on TTFT.
# 512/rank is past the knee, where scheduling more chunks costs more than the
# smaller chunk saves.
export CHUNK="${CHUNK:-8192}"

# EAGLE draft depth. TPOT = decode step time / accept_length, and each extra draft
# step measured 6.53 ms here, so deepening pays only while acceptance rises faster
# than the step cost. 5/1/6 accepts 4.64 against a 4.00 break-even; 7/1/8 both
# misses its break-even and runs prefill out of activation memory. Both legs must
# agree -- SGLang rejects a disagg pair whose speculative config differs.
export MTP="${MTP:-1}"                            # 0 disables speculative decoding
export MTP_STEPS="${MTP_STEPS:-5}"
export MTP_TOPK="${MTP_TOPK:-1}"
export MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-6}"

# Prometheus on both engines, off by default in SGLang. Without it /metrics 404s,
# and that endpoint is the only network-reachable source of the MTP acceptance
# length verify.sh checks. --enable-metrics-for-all-schedulers matters under
# dp-attention: TP 0 is the only scheduler that reports otherwise, so 1 of 8 DP
# ranks would stand in for the fleet.
export ENGINE_METRICS="${ENGINE_METRICS:-1}"

# --- router -----------------------------------------------------------------
# rust execs the infera-router binary the image carries at /usr/local/bin. It
# makes the same routing decisions as the python backend request for request and
# measured 27% faster end to end, so it is the default. Its supported subset is
# what this example passes anyway (etcd discovery, http transport, kv-aware).
export ROUTER_BACKEND="${ROUTER_BACKEND:-rust}"
export ROUTER_POLICY="${ROUTER_POLICY:-kv-aware}"

# --- kvd: KV offload below the GPU cache ------------------------------------
# KVD=1 runs an infera-kvd daemon beside the prefill engine and points SGLang's
# hierarchical cache at it, so prefixes evicted from the GPU survive in host RAM
# (L2) and on local NVMe (L3).
#
# KVD=0 is the default because the tier measured SLOWER on the workload this
# recipe was tuned on: it cost 12% and served zero reads, the GPU pool alone
# already covering ~100% of the reuse that trace had to offer. Turn it on when
# your working set outgrows 54 GB/rank; see README §6 for how to tell.
#
# Prefill leg only. SGLang issues storage prefetch on its aggregated and prefill
# branches, never on the decode branch, so kvd on a decode leg is write-only --
# infera refuses to wire it there and says so in the log.
export KVD="${KVD:-0}"
export KVD_SOCKET="${KVD_SOCKET:-/tmp/infera-kvd/kvd.sock}"
# Must be node-local NVMe. Anything shared (NFS, weka) classifies as buffered and
# the reload lands in the TTFT budget instead of under it.
export KVD_L3_DIR="${KVD_L3_DIR:-/your/path/kvd-l3}"
export KVD_RAM_BYTES="${KVD_RAM_BYTES:-64G}"      # L2 arena, pinned host RAM
export KVD_LONG_BYTES="${KVD_LONG_BYTES:-512G}"   # L3 budget under KVD_L3_DIR
# O_DIRECT vs buffered for L3. `auto` classifies the mount by walking sysfs from
# its major:minor, which is not namespaced and so works unprivileged in a
# container; it falls back to `buffered` whenever it cannot identify the device,
# an overlay path being the usual reason. Pin `direct` when you know KVD_L3_DIR
# is local NVMe: a misclassification is a silent 4x, 14.56 GB/s against 3.70 on
# the LVM-over-NVMe xfs this was measured on. launch_kvd.sh prints the verdict.
export KVD_IO_MODE="${KVD_IO_MODE:-auto}"
# One tablespace slot must hold one whole hicache page, and a page holds every
# layer: GLM-5.2-FP8 at page_size 64 writes 2.74 MiB per KV page and 624 KiB per
# DSA-indexer page. Two pools cover both without wasting a slot on the smaller.
# A value that outgrows its largest pool is REJECTED, not split, which leaves L3
# silently empty -- verify.sh fails on that rather than letting you find it later.
export KVD_TABLESPACE_POOLS="${KVD_TABLESPACE_POOLS:-1M,4M}"
# SGLang's own host tier, in GB PER DP RANK (so x8 here). It sits between the GPU
# pool and kvd and stages L3 reads. Deliberately smaller than the 54 GB device
# pool per rank: matching it would pin ~870 GB of host RAM for an L2 that kvd's L3
# already backs.
export HICACHE_SIZE="${HICACHE_SIZE:-32}"
# KVD=true is not "1": the daemon would be skipped, the engine would start with
# no tier, and the run would look like a kvd run. Refuse rather than guess.
case "$KVD" in 0|1) ;; *) echo "[env] KVD='$KVD' is not '0' or '1'" >&2; exit 1 ;; esac

# --- bench sizing -----------------------------------------------------------
# Deliberately small: bench.sh sizes the deployment, it is not a sweep. Four waves
# of 16 concurrent is a few minutes. NUM_PROMPTS follows CONC so that raising the
# concurrency alone keeps the same number of waves. See README §5.
export ISL="${ISL:-4096}"
export OSL="${OSL:-1024}"
export CONC="${CONC:-16}"
export NUM_PROMPTS="${NUM_PROMPTS:-$((CONC * 4))}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" 2>/dev/null || true
