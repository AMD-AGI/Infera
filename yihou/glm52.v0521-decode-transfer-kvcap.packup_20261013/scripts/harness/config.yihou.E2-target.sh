#!/usr/bin/env bash
# E2 "fixed target": the phase-1 target recipe plus the candidate H1 fix.
# Each delta is ":=" so a KEY=VALUE override still wins; everything else comes from
# config.yihou.v0521-base.sh (requires YIHOU_NODE / YIHOU_IP / YIHOU_JOB / YIHOU_ROUND).
YIHOU_EXP=E2

# H1 (HiCache host registration): fixed 450 GB/rank pool, but registered in ONE
#   cudaHostRegister call: SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512 (> 450) on the
#   prefill leg (sglang default 256 -> two calls for 450 GB).
: "${PREFILL_HICACHE_SIZE:=450}"
: "${PREFILL_HICACHE_RATIO:=1.5}"     # still emitted by engine.sh; overridden by size > 0
: "${HICACHE_HOST_REGISTER_CHUNK_GB:=512}"

# H2 (index_share): index_share_for_mtp_iteration=false.
: "${INDEX_SHARE_FOR_MTP:=false}"

# H3 (gfx950 fused DSA indexer): v0521 default = ON (no flag, no env).
: "${DSA_FUSED_INDEXER:=default}"
: "${AITER_FUSED_FP8_INDEXER_WRITER:=default}"

# H8 (decode memory headroom): decode mem-fraction 0.95 (the target).
: "${DECODE_MEM_FRACTION:=0.95}"

# Load: CONC=44 (the target).
: "${CONC:=44}"

source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.v0521-base.sh"
