#!/usr/bin/env bash
# Overlay: sglang PR #42553 (head 8c27ef48) — HiCache host pools from one exact-size
# hipHostMalloc (GTT) instead of mmap + chunked hipHostRegister (USERPTR). Opt-in, PREFILL only.
#
# Use:  export CONFIG=$H/config.yihou.gtt42553.sh
#       export GTT_BASE_CONFIG=$H/config.yihou.E1-relaxed.sh   # or config.yihou.mc4340.sh (+ MC4340_BASE_CONFIG)
#
# What it changes vs the base chain (and nothing else):
#   1. mounts pool_host/common.py = v0521 image file (sha256 781e2481..., same in 1009 and
#      mc4340 images) + the PR's common.py hunk (diag/pr42553/);
#   2. PREFILL env HSA_USERPTR_FOR_PAGED_MEM=0 (the PR's opt-in; process-wide ROCr switch,
#      so it is NOT set on decode, which has no HiCache).
# With it, H1 (chunked registration) cannot occur: _cuda_host_register is bypassed.
#
# HARD LIMIT: GTT is the TTM global page limit = MemTotal/2 = ~1477 GB on 102/136, shared by
# ALL ranks on the node. 4 prefill ranks × (KV pool + DSA indexer pool, indexer ≈ 0.229 × KV)
# must fit, or the PR raises at startup (no fallback). 450 GB target does NOT fit (≈2.21 TB).

_gtt_base="${GTT_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.E1-relaxed.sh}"
[[ -r "$_gtt_base" ]] || { echo "config.yihou.gtt42553.sh: GTT_BASE_CONFIG not readable: $_gtt_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_gtt_base"

case "$YIHOU_MOUNTS" in
    "$WS/harness/mounts.yihou.v2.tsv")         YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-gtt.tsv" ;;
    "$WS/harness/mounts.yihou.v2-regbase.tsv") YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-regbase-gtt.tsv" ;;
    *) echo "config.yihou.gtt42553.sh: no GTT variant for YIHOU_MOUNTS=$YIHOU_MOUNTS" >&2
       return 2 2>/dev/null || exit 2 ;;
esac

# Budget guard (explicit --hicache-size only; ratio sizing ≤ 2 is ~≤ 200 GB/rank on this shape).
: "${GTT_TOTAL_GB:=1477}"
if (( PREFILL_HICACHE_SIZE > 0 )); then
    _gtt_need=$(( PREFILL_HICACHE_SIZE * 4 * 1229 / 1000 ))
    if (( _gtt_need * 100 > GTT_TOTAL_GB * 95 )); then
        echo "config.yihou.gtt42553.sh: --hicache-size $PREFILL_HICACHE_SIZE needs ~${_gtt_need} GB GTT > 95% of ${GTT_TOTAL_GB} GB" >&2
        return 2 2>/dev/null || exit 2
    fi
elif [[ "${PREFILL_HICACHE_RATIO%%.*}" -gt 2 ]]; then
    echo "config.yihou.gtt42553.sh: hicache ratio $PREFILL_HICACHE_RATIO not validated against the GTT budget" >&2
    return 2 2>/dev/null || exit 2
fi

PREFILL_EXTRA_ENV+=" HSA_USERPTR_FOR_PAGED_MEM=0"
# CHUNK_GB is inert on this path; leave it set so a GTT-off rerun keeps the H1 fix.
YIHOU_EXP="${YIHOU_EXP:-E}+gtt42553"
unset _gtt_base _gtt_need
