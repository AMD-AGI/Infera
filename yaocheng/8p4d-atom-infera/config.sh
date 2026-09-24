#!/usr/bin/env bash
# GLM-5.2 MXFP4 8P4D on native ATOM + Infera, Slurm job 31626.
# Sourced by scripts/common.sh; KEY=VALUE arguments override every default.
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$KIT_DIR/../.." && pwd)"
TMP_DIR="${TMP_DIR:-$KIT_DIR/.tmp}"
RESULTS_DIR="${RESULTS_DIR:-$KIT_DIR/results}"

# Nodes. HTTP, etcd and ZMQ use the fenic addresses; KV moves over ionic RDMA.
PREFILL_NODE="${PREFILL_NODE:-smci355-ccs-aus-n04-25}"
PREFILL_IP="${PREFILL_IP:-10.235.192.131}"
DECODE_NODE="${DECODE_NODE:-smci355-ccs-aus-n04-29}"
DECODE_IP="${DECODE_IP:-10.235.192.57}"
# etcd, router, AgentX client and the image build run on the control node,
# which is also where the scripts are executed.
CONTROL_NODE="${CONTROL_NODE:-$DECODE_NODE}"
CONTROL_IP="${CONTROL_IP:-$DECODE_IP}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"

IMAGE_BASE="${IMAGE_BASE:-rocm/atom-dev:nightly_202609221542}"
IMAGE="${IMAGE:-infera-atom:nightly_202609221542}"
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.14}"
MODEL="${MODEL:-/apps/data/models/GLM-5.2-MXFP4}"
PREFIX="${PREFIX:-glm52-8p4d-atom}"

# Listening ports stay below the ephemeral range (32768-60999) and outside
# Mooncake's random RPC ports (15000-17000); outgoing sockets can take a port
# there before the listener binds.
ETCD_PORT="${ETCD_PORT:-23379}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-23380}"
ROUTER_PORT="${ROUTER_PORT:-18000}"
PREFILL_PORT="${PREFILL_PORT:-19001}"
DECODE_PORT="${DECODE_PORT:-19002}"
# Prefill DP rank r listens on PREFILL_HANDSHAKE_PORT + r.
PREFILL_HANDSHAKE_PORT="${PREFILL_HANDSHAKE_PORT:-21301}"
DECODE_HANDSHAKE_PORT="${DECODE_HANDSHAKE_PORT:-21311}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"

# Engines. CONC sizes the decode batch and picks the MTP depth; depth and forced
# acceptance move together (golden AL: K4 -> 3.33, K3 -> 2.99).
# MTP_AL= (empty) turns forced acceptance off for correctness checks.
CONC="${CONC:-16}"
if (( CONC >= 48 )); then MTP_K="${MTP_K:-3}"; MTP_AL="${MTP_AL-2.99}"
else MTP_K="${MTP_K:-4}"; MTP_AL="${MTP_AL-3.33}"; fi
# Prefill graph mode. MTP decode steps that run eager need
# patch/atom-moe-eager-decode-pad.diff (DP MoE gather padding).
PREFILL_GRAPH_ARGS="${PREFILL_GRAPH_ARGS:---enforce-eager}"
PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3,4,5,6,7}"
DECODE_GPUS="${DECODE_GPUS:-0,1,2,3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Experts of layers 0-77 stay MXFP4 and the MTP layer 78 stays BF16.
QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*","model.layers.78.*"]}'
ENGINE_ENV=(
    AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_USE_FLYDSL_MOE_SORTING=1
    AITER_LOG_LEVEL=WARNING PYTHONHASHSEED=0 ATOM_MLA_PAGE_SIZE=1
    ATOM_ONLINE_QUANT_STREAMING=0 ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047
    ATOM_USE_TRITON_MLA=0 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_GID_INDEX=1
    INFERA_ATOM_READY_TIMEOUT="$READY_TIMEOUT"
    TRITON_CACHE_DIR=/root/.cache/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
)
# The fabric routes same-numbered rails only. Each decode rank registers its own
# ionic_<gpu>; the prefill keeps one engine per rail and writes on the rail of
# the decode rank it serves. Session affinity keeps every turn of an agent
# session (X-Correlation-ID) on one prefill DP rank, whose prefix cache it reuses.
PREFILL_ENV="ATOM_MOONCAKE_MATCHED_RAILS=auto ATOM_DP_SESSION_AFFINITY=1"
DECODE_ENV=""
PREFILL_EXTRA_ENV="${PREFILL_EXTRA_ENV:-}"    # space-separated KEY=VALUE
DECODE_EXTRA_ENV="${DECODE_EXTRA_ENV:-}"
PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS:-}"
DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS:-}"

# AgentX. Same InferenceX commit and concurrency points as
# ../2p1d-sweep-triton-dsa-20260922 so the two result sets compare.
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-918524ff94045b3f091115f1051c22a8588edf2b}"
DURATION="${DURATION:-3600}"
WARMUP_PER_LANE="${WARMUP_PER_LANE:-10}"
POINTS="${POINTS:-80 112 144 192 256}"
