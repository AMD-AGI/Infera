#!/usr/bin/env bash
# E1 "relaxed baseline": phase-1 recipe with every suspect knob moved to its safe side.
# If E1 passes AgentX full and E2 does not, the culprit is among the deltas below.
# Each delta is ":=" so a KEY=VALUE override still wins; everything else comes from
# config.yihou.v0521-base.sh (requires YIHOU_NODE / YIHOU_IP / YIHOU_JOB / YIHOU_ROUND).
YIHOU_EXP=E1

# H1 (HiCache host registration): ratio sizing instead of a fixed 450 GB pool.
#   --hicache-ratio 1.5 and NO --hicache-size -> ~154 GB/rank, below the 256 GB
#   per-call cudaHostRegister chunk limit, so ONE registration per pool.
: "${PREFILL_HICACHE_SIZE:=0}"
: "${PREFILL_HICACHE_RATIO:=1.5}"
: "${HICACHE_HOST_REGISTER_CHUNK_GB:=}"

# H2 (index_share): index_share_for_mtp_iteration=false (same as phase-1 c44d/c44e).
: "${INDEX_SHARE_FOR_MTP:=false}"

# H3 (gfx950 fused DSA indexer): OFF on both legs, both switches
#   (--no-enable-dsa-fused-indexer AND SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1).
: "${DSA_FUSED_INDEXER:=off}"
: "${AITER_FUSED_FP8_INDEXER_WRITER:=off}"

# H8 (decode memory headroom): decode mem-fraction 0.85 instead of 0.95.
: "${DECODE_MEM_FRACTION:=0.85}"

# Load: CONC=40 (the reference C40-full point) instead of 44.
: "${CONC:=40}"

source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.v0521-base.sh"
