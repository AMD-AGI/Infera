#!/usr/bin/env bash
# ============================================================================
#  Site wrapper — vultr MI355X, chi2835. THE ONLY FILE CARRYING SITE VALUES.
# ============================================================================
# Everything here is either copied from a previous verified run on this cluster
# (../infera/verify_example.packup_20260804) or measured. Nothing is guessed.
#
# Usage (ON the node):  bash mix_site.sh up | smoke | down
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- node ------------------------------------------------------------------
export MY_IP="${MY_IP:-10.2.122.78}"      # chi2835 enp193s0f1np1 — data plane, NOT 45.76.x mgmt
export ETCD_IP="$MY_IP"                   # mix: etcd is on this same node

# ---- image and weights -----------------------------------------------------
export INFERA_IMAGE="${INFERA_IMAGE:-infera/engine-sglang:merged-e}"
export MODEL_MOUNT="/mnt/vast"
export MODEL="/mnt/vast/xiaobo/models/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# This image's ENTRYPOINT injects the HOST's libionic so in-container libibverbs speaks the
# host ionic_rdma kmod's ABI. MIX moves no KV over the wire, so RDMA is not load-bearing
# here — but keeping the entrypoint costs nothing and avoids a difference vs the PD runs.
export ENTRYPOINT_KEEP=0

# ---- deployment shape (starting point; the A/B knobs are TP/DPA/CHUNK/GMU) --
export CTR=glm52_mix
export TP=8
export CTX=262144
export CHUNK=65536        # GLOBAL budget; SGLang divides by dp_size only under DPA
export GMU=0.80           # ONE pool for both phases (PD used 0.70 prefill / 0.85 decode)
export MAX_RUNNING=256
export CUDA_GRAPH_BS=128

export DPA=1
export MTP=1
export KVAWARE=1
export KVD=1
export HICACHE_GB=32

export ROUTER_POLICY=kv-aware
export ROUTER_BACKEND=rust

case "${1:-up}" in
  up)    exec bash "$DIR/mix_up.sh" ;;
  smoke) exec bash "$DIR/mix_smoke.sh" ;;
  down)  source "$DIR/mix_common.sh"; reap; docker rm -f "$CTR" "$ETCD_CTR" 2>/dev/null; exit 0 ;;
  *)     echo "usage: bash mix_site.sh up|smoke|down" >&2; exit 2 ;;
esac
