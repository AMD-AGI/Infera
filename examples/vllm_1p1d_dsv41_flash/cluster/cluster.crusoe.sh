#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Wrapper for crusoe spur (crsuse2-m2m-v3-* MI355X VMs, amdgpu-dkms 6.16.13, 8x Pensando ionic 400G),
# one node: prefill on GPUs 0,1 and decode on GPUs 4,5 unless PREFILL_GPUS / DECODE_GPUS name the
# GPUs the job was given. Run it ON the node (inside the job).
#
# What differs from cluster.gem.sh, and why:
# - GEM KV (MR_MODE=gem): 6.16.13 charges dma-buf-pinned KFD VRAM twice and these hosts have no
#   peer memory (ibv_reg_mr on VRAM fails), so dma-buf on GEM BOs is the only path charged once.
# - One NIC for every GPU of both legs: each ionic is its own routed rail and ionic_i only reaches
#   ionic_i. Any prefill rank sends to any decode rank, and the image's Mooncake (PyPI build) has
#   no destination-NIC write selection, so a per-GPU NIC map would pair NICs on different rails.
#   All KV loops back through the NIC of the first decode GPU (ionic_k for GPU k), so the NIC
#   writing into decode memory sits on the decode socket.
# - The engine container never mounts the host libionic (it needs GLIBC 2.38, the image has 2.35);
#   the image's own provider drives these NICs. common.sh already does not mount it.
#
# Usage:  bash cluster/cluster.crusoe.sh up | smoke | accuracy | golden | block | agentx CONC... | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PREFILL_NODE="${PREFILL_NODE:-$(hostname -s)}"
export DECODE_NODE="${DECODE_NODE:-$PREFILL_NODE}"
export PREFILL_IP="${PREFILL_IP:-$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')}"
export DECODE_IP="${DECODE_IP:-$PREFILL_IP}"
export KIT_DIR="$KIT"
export PREFILL_GPUS="${PREFILL_GPUS:-0,1}"
export DECODE_GPUS="${DECODE_GPUS:-4,5}"

export INFERA_IMAGE="${INFERA_IMAGE:-<image from build_image.sh>}"
# Weights staged to the node's NVMe: a cold read from /shared_nfs takes several minutes per engine.
export MODEL_MOUNT="${MODEL_MOUNT:-/mnt/m2m_nobackup/liyingli_dsv41/models}"
export MODEL="${MODEL:-$MODEL_MOUNT/DeepSeek-V4.1-Flash}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export LOG_DIR="${LOG_DIR:-/tmp/dsv41_pd}"

export MR_MODE=gem
LOOP_NIC="${LOOP_NIC:-ionic_${DECODE_GPUS%%,*}}"
export RDMA_IB_DEVICES="${RDMA_IB_DEVICES:-$LOOP_NIC}"
export RDMA_GPU_NICS="${RDMA_GPU_NICS:-$(printf "$LOOP_NIC,%.0s" 1 2 3 4 5 6 7 8 | sed 's/,$//')}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"
export MC_ENV_EXTRA="${MC_ENV_EXTRA:-MC_IB_TIMEOUT=18}"

export UMBP="${UMBP:-1}" UMBP_GB="${UMBP_GB:-449}" UMBP_MODE="${UMBP_MODE:-standalone}"
# Off the common defaults (sglang and the kit both default to 30000): a 4-GPU job shares the node.
export ETCD_PORT="${ETCD_PORT:-22379}" ROUTER_PORT="${ROUTER_PORT:-28700}"
export PREFILL_PORT="${PREFILL_PORT:-31300}" DECODE_PORT="${DECODE_PORT:-31301}"

export INFERENCEX_DIR="${INFERENCEX_DIR:-<SemiAnalysisAI/InferenceX checkout>}"
export AGENTX_HF_HOME="${AGENTX_HF_HOME:-/mnt/m2m_nobackup/liyingli_dsv41/hf}"
export AGENTX_RUNTIME_DIR="${AGENTX_RUNTIME_DIR:-/mnt/m2m_nobackup/liyingli_dsv41/agentx-runtime}"
# AIPerf's warmup holds the replay until every lane's cold first request returns; past c704 that first wave (one
# ~128k-token cold prompt per lane) can outlast 1800 s on these nodes and AIPerf then voids the run. InferenceX's own
# recipes use 1800 or 3600; the scored window is the same either way.
export WARMUP_GRACE="${WARMUP_GRACE:-3600}"

# Passive decode-hang capture alongside every AgentX point (engine/hang_watch.sh); HANG_WATCH=0 turns it off.
if [ "${1:-}" = agentx ] && [ "${HANG_WATCH:-1}" = 1 ]; then
  nohup bash "$KIT/engine/hang_watch.sh" "$LOG_DIR" "${CTR:-dsv41_pd}" "$DECODE_PORT" > "$LOG_DIR/hang-watch.log" 2>&1 &
fi

exec bash "$KIT/cluster/run.sh" "$@"
