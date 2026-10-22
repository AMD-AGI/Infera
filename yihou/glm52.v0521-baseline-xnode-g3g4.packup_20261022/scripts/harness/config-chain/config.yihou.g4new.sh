#!/usr/bin/env bash
# Overlay (g4tune 2026-10-20): the user's standing rule for ALL experiments from 2026-10-20 on.
#   1. sglang PR #42553 (merged, head b2de502825): HiCache host pools from one exact-size hipHostMalloc (GTT)
#      instead of hipHostRegister (USERPTR). Mount mounts-v0521/src/pool_host_common.yihou-pr42553.py
#      (= image common.py sha256 781e2481 + the PR's final common.py hunk; sha256 e35b267e, identical to the
#      8c27ef48-based file built 10-10) and PREFILL env HSA_USERPTR_FOR_PAGED_MEM=0 (the PR's opt-in; process-wide,
#      so decode, which has no HiCache, does not get it).
#   2. index_share_for_mtp_iteration=true (INDEX_SHARE_FOR_MTP=true; was false since c44d, 10-09).
#   3. decode WITHOUT --disable-custom-all-reduce (was the garbled-MTP fix; correctness is waived by sim 3.61).
#   (DSA backends stay triton/triton on both legs, as in the base chain.)
# Bisect knobs (2026-10-20, after decode Memory access faults in g4n2/g4n2b/g4n1): G4NEW_IDXSHARE true|false
# (default true) and G4NEW_CUSTOM_AR on|off (default on; off = keep decode --disable-custom-all-reduce).
# Chain: NEW_BASE_CONFIG (default config.yihou.xferfix2.sh, which needs XFERFIX_BASE_CONFIG).
# GTT guard: all 4 prefill ranks' HiCache KV pool + DSA indexer host pool (indexer = 0.229 x KV, measured g4xb5:
# 450 GB -> 103.13 GB) come out of GTT = mem_info_gtt_total (1477 GB on m2m-097/250). The PR raises at startup
# if it does not fit, so refuse here first. 450 GB/rank needs ~2212 GB.
: "${G4NEW_IDXSHARE:=true}"; : "${G4NEW_CUSTOM_AR:=on}"
case "$G4NEW_IDXSHARE:$G4NEW_CUSTOM_AR" in true:on|true:off|false:on|false:off) ;;
  *) echo "config.yihou.g4new.sh: bad G4NEW_IDXSHARE/G4NEW_CUSTOM_AR" >&2; return 2 2>/dev/null || exit 2 ;; esac
INDEX_SHARE_FOR_MTP="$G4NEW_IDXSHARE"
_g4n_base="${NEW_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.xferfix2.sh}"
[[ -r "$_g4n_base" ]] || { echo "config.yihou.g4new.sh: base not readable: $_g4n_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_g4n_base"
[[ "$INDEX_SHARE_FOR_MTP" == "$G4NEW_IDXSHARE" ]] || { echo "config.yihou.g4new.sh: INDEX_SHARE_FOR_MTP was overridden by the chain" >&2; return 2 2>/dev/null || exit 2; }

case "$YIHOU_MOUNTS" in
    "$WS/harness/mounts.yihou.v2-xferfix2.tsv") YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix2-gtt.tsv" ;;
    *) echo "config.yihou.g4new.sh: no GTT variant for YIHOU_MOUNTS=$YIHOU_MOUNTS" >&2; return 2 2>/dev/null || exit 2 ;;
esac

if [[ "$G4NEW_CUSTOM_AR" == on ]]; then
    DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS//--disable-custom-all-reduce/}"
    PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS//--disable-custom-all-reduce/}"
    [[ " $DECODE_EXTRA_ARGS $PREFILL_EXTRA_ARGS " == *"--disable-custom-all-reduce"* ]] && {
        echo "config.yihou.g4new.sh: --disable-custom-all-reduce still present" >&2; return 2 2>/dev/null || exit 2; }
else
    [[ " $DECODE_EXTRA_ARGS " == *" --disable-custom-all-reduce "* ]] || {
        echo "config.yihou.g4new.sh: G4NEW_CUSTOM_AR=off but decode lacks --disable-custom-all-reduce" >&2; return 2 2>/dev/null || exit 2; }
fi

: "${GTT_TOTAL_GB:=1477}"
(( PREFILL_HICACHE_SIZE > 0 )) || { echo "config.yihou.g4new.sh: needs an explicit PREFILL_HICACHE_SIZE (GB/rank)" >&2; return 2 2>/dev/null || exit 2; }
_g4n_need=$(( PREFILL_HICACHE_SIZE * 4 * 1229 / 1000 ))
if (( _g4n_need * 100 > GTT_TOTAL_GB * 95 )); then
    echo "config.yihou.g4new.sh: --hicache-size $PREFILL_HICACHE_SIZE needs ~${_g4n_need} GB GTT > 95% of ${GTT_TOTAL_GB} GB" >&2
    return 2 2>/dev/null || exit 2
fi
PREFILL_EXTRA_ENV+=" HSA_USERPTR_FOR_PAGED_MEM=0"
YIHOU_EXP="${YIHOU_EXP:-E}+pr42553-idx$G4NEW_IDXSHARE-car$G4NEW_CUSTOM_AR"
unset _g4n_base _g4n_need
