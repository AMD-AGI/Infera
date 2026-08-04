#!/usr/bin/env bash
# Site wrapper — vultr MI355X cluster, chi2835 (prefill) + chi2879 (decode).
#
# Derived from cluster/cluster.peermem.sh. Both nodes report:
#     peermem : present (module:ib_peer_mem)
#     mode A  : VIABLE * best   bare ibv_reg_mr + peer-mem
# so this is the mode-A wrapper. Every transport value below is COPIED from the
# preflight report, not guessed — see notes/preflight_*.txt.
#
# Usage:  bash cluster.vultr.sh up | smoke | bench [conc...] | down
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
export PREFILL_NODE="chi2835"
export DECODE_NODE="chi2879"
export PREFILL_IP="10.2.122.78"      # enp193s0f1np1 — data plane, NOT the 45.76.x mgmt NIC
export DECODE_IP="10.2.122.10"       # enp193s0f1np1
export KIT_DIR="$KIT"                # /mnt/vast/... — same path on both nodes (shared VAST)

# ---------------------------------------------------------------------------
# 2. Image and weights
# ---------------------------------------------------------------------------
# The kit's default tag `inferaimage/infera-sglang:0.2.0` is a placeholder and
# does not exist on this cluster. This is the image both nodes carry.
export INFERA_IMAGE="${INFERA_IMAGE:-infera/engine-sglang:merged-e}"

export MODEL_MOUNT="/mnt/vast"
export MODEL="/mnt/vast/xiaobo/models/GLM-5.2-MXFP4"
export TOKENIZER="$MODEL"

# This image's ENTRYPOINT (/usr/local/bin/infera-inject-host-ionic) copies the
# HOST's libionic over the container's, so the in-container libibverbs speaks the
# host ionic_rdma kmod's ABI. Skipping it leaves the provider mismatched and
# mooncake degrades silently. Pass the SYMLINK, not the resolved file: docker
# resolves the chain at mount time, and the two nodes carry different builds
# (chi2835 -184, chi2879 -187).
export HOST_RDMA_LIB=/usr/lib/x86_64-linux-gnu/libionic.so.1
export HOST_RDMA_MOUNT=/host-libionic/libionic.so   # the path THIS image's entrypoint reads
export ENTRYPOINT_KEEP=1

# ---------------------------------------------------------------------------
# 3. Transport — mode A (peer-mem, every rail)
# ---------------------------------------------------------------------------
# Preflight rail lists DIFFER between the two nodes:
#     chi2835  ionic_0..7            (8 active, 3200 Gb/s)
#     chi2879  ionic_0..4,6,7        (ionic_5 PORT_DOWN, 7 active, 2800 Gb/s)
# The kit exposes ONE global RDMA_IB_DEVICES, so this is their intersection —
# every device listed is PORT_ACTIVE on BOTH nodes, which is what the kit's own
# rule requires. See notes/kit_findings.md.
export RDMA_IB_DEVICES="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_6,ionic_7"

# Both nodes resolved the routable RoCEv2 GID at index 1.
export MC_GID_INDEX=1

export MOONCAKE_DISABLE_HIP_DMABUF=1   # peer-mem present -> bare ibv_reg_mr is the no-pin path
export RDMAV_FORK_SAFE=1
# MC_MS_FILTERS deliberately unset: every rail may carry KV.

# ---------------------------------------------------------------------------
# 4. Deployment shape — kit defaults, unchanged
# ---------------------------------------------------------------------------
export TP=8
export CTX=262144
export CHUNK=65536

export PREFILL_DPA=0
export DECODE_DPA=1
export PREFILL_MTP=0
export DECODE_MTP=1

export ROUTER_POLICY=kv-aware
export KVAWARE=1
export PREFILL_KVD=1
export DECODE_KVD=0

export GMU_PREFILL=0.70
export GMU_DECODE=0.85

exec bash "$KIT/engine/${1:-up}.sh" "${@:2}"
