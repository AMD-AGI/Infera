#!/usr/bin/env bash
# Purpose: the config.full.sh target, adapted to this run's 1P1D P4D4 shape on
#   crsuse2-m2m-135 (prefill) + crsuse2-m2m-138 (decode), for AgentX CONC=40.
# Usage: pass CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv to the
#   bench scripts. Everything not set here is inherited from config.full.sh.
# Artifacts: none; this file only defines shell variables.
#
# Relationship to config.yihou.sh: identical hardware/topology/same-rail
# settings, but sources config.full.sh instead of config.sh, so it picks up
# Prefill HiCache, max-running/graph BS 128, JIT grouped-topk, warmup lane 10,
# 3600 s profiling, and the removal of the IndexShare workaround.

# Set before sourcing: config.full.sh uses ${VAR:-default} throughout, so these
# win, and a KEY=VALUE argument on the command line still wins over these.

: "${IMAGE:=infera-sglang:v0519-yihou-0917}"
: "${SGLANG_BASE_IMAGE:=lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917}"

: "${CONTROL_NODE:=crsuse2-m2m-135}"
: "${BUILDER_NODE:=crsuse2-m2m-135}"
: "${CONTAINER_PREFIX:=glm52-pd-yihou-1p1d-full}"

# ---------------------------------------------------------------------------
# SUBSTITUTIONS against config.full.sh — the two unported TODO items
# ---------------------------------------------------------------------------
# config.full.sh requests DSA backends that the pinned nightly does not accept.
# Verified first-hand inside infera-sglang:v0519-yihou-0917, in
# sglang/srt/arg_groups/fields/{exec_,spec}.py:
#
#   dsa_prefill_backend choices = flashmla_sparse, flashmla_sparse_q8,
#       flashmla_kv, flashmla_auto, flashinfer_sparse_mla, fa3, tilelang,
#       triton, aiter, trtllm        -> no "flydsl"
#   dsa_topk_backend    choices = sgl-kernel, torch, flashinfer
#                                    -> no "aiter"
#
# engine.sh forwards both (lines 148-149, 165-166), so config.full.sh's values
# would be rejected by argparse and the engine would die at startup. This
# matches issue.md 2.1 / 2.2, which list both as unported TODOs requiring AITER
# and SGLang kernel work, not a flag flip.
#
# So: fall back to engine.sh's own default backend, and leave the top-k backend
# unset so engine.sh omits the flag entirely and sglang uses sgl-kernel.
# EVERYTHING ELSE IS config.full.sh AS WRITTEN.
: "${DSA_PREFILL_BACKEND:=tilelang}"
: "${DSA_DECODE_BACKEND:=tilelang}"
# DSA_TOPK_BACKEND is cleared AFTER the source below, not here: config.full.sh
# assigns it with ${DSA_TOPK_BACKEND:-aiter}, and ":-" treats an empty string as
# unset, so clearing it beforehand silently yields "aiter" again.

# ---------------------------------------------------------------------------
# Hardware shape — same as config.yihou.sh, same reasoning
# ---------------------------------------------------------------------------
# GPU[1] on crsuse2-m2m-135 is held by a root-owned Kubernetes pod
# (pod5d84e491, "vllm serve Qwen3-32B") and ionic_7 on that node is defective
# (no netdev, GID index 1 all-zero). Devices 2,3,4,5 avoid both.
: "${PREFILL_GPU_DEVICES:=2,3,4,5}"
: "${DECODE_GPU_DEVICES:=2,3,4,5}"
: "${PREFILL_TP:=4}"
: "${PREFILL_DP:=4}"
: "${DECODE_TP:=4}"
: "${DECODE_DP:=4}"

# ---------------------------------------------------------------------------
# Same-rail KV transfer — both halves, as in config.yihou.sh
# ---------------------------------------------------------------------------
: "${PD_DP_RANK_AFFINITY:=1}"

# Per-GPU HCA pinning. The shared-list form leaves HCA choice to Mooncake
# auto-discovery, which lets the two ends pick different NICs; these rails are
# physically isolated, so a mismatch is unreachable. Keys are the LOCAL
# (visible) device index. Assigned with an explicit test because a bare closing
# brace inside ${VAR:=...} truncates the JSON.
if [[ -z "${RDMA_DEVICE:-}" ]]; then
    RDMA_DEVICE='{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}'
fi
# MC_TE_FILTERS defaults to $RDMA_DEVICE upstream, which would be a JSON blob
# where a device list is expected.
: "${MC_TE_FILTERS:=ionic_2,ionic_3,ionic_4,ionic_5}"

# Inherit the full target: Prefill HiCache on, max-running and graph max BS 128,
# SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1, warmup 10 requests per lane,
# AGENTX_DURATION=3600, and JSON_MODEL_OVERRIDE_ARGS emptied (the
# index_share_for_mtp_iteration workaround removed).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.full.sh"

# Cleared post-source (see the note above): engine.sh only passes
# --dsa-topk-backend when this is non-empty, so an empty value makes sglang use
# its own default, sgl-kernel. "aiter" is not an accepted choice on this base.
DSA_TOPK_BACKEND=""
