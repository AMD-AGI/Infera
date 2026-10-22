#!/usr/bin/env bash
# Overlay (idxshare debug E2, 2026-10-20): E1 (index_share true + decode debug agent + seed guard count) plus the
# top-k length guard in the indexer (dsa_indexer.yihou-topkguard.py = image dsa_indexer.py + YIHOU-TOPKGUARD:
# per forward-mode/eager-graph counts of rows whose top-k length is < 0 or > the logits width; "mask" clamps them).
#   E2_BASE_CONFIG     chain to source (default config.yihou.dbgE1.sh)
#   YIHOU_TOPK_GUARD   off|count|mask (default count) -> DECODE env only
_e2_base="${E2_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.dbgE1.sh}"
[[ -r "$_e2_base" ]] || { echo "config.yihou.dbgE2.sh: base not readable: $_e2_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_e2_base"
case "$YIHOU_MOUNTS" in
    "$WS/harness/mounts.yihou.v2-xferfix2-gtt-seedguard.tsv") YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix2-gtt-seedguard-topkguard${YIHOU_TOPKGUARD_VER:-}.tsv" ;;
    *) echo "config.yihou.dbgE2.sh: unexpected YIHOU_MOUNTS=$YIHOU_MOUNTS" >&2; return 2 2>/dev/null || exit 2 ;;
esac
: "${YIHOU_TOPK_GUARD:=count}"
case "$YIHOU_TOPK_GUARD" in off|count|mask) ;; *)
    echo "config.yihou.dbgE2.sh: YIHOU_TOPK_GUARD=$YIHOU_TOPK_GUARD (want off|count|mask)" >&2
    return 2 2>/dev/null || exit 2 ;; esac
DECODE_EXTRA_ENV+=" YIHOU_TOPK_GUARD=$YIHOU_TOPK_GUARD"
# YIHOU_SKGUARD (optional): off|count|mask for the sgl_kernel/top_k.py call-site guard only (else it follows
# YIHOU_TOPK_GUARD). mask clamps lengths into the score row AFTER logging a BAD call.
if [[ -n "${YIHOU_SKGUARD:-}" ]]; then
    case "$YIHOU_SKGUARD" in off|count|mask) ;; *) echo "config.yihou.dbgE2.sh: YIHOU_SKGUARD=$YIHOU_SKGUARD" >&2; return 2 2>/dev/null || exit 2 ;; esac
    DECODE_EXTRA_ENV+=" YIHOU_SKGUARD=$YIHOU_SKGUARD"
fi
YIHOU_EXP="${YIHOU_EXP:-E}+topkguard-$YIHOU_TOPK_GUARD${YIHOU_SKGUARD:+-sk$YIHOU_SKGUARD}"
unset _e2_base
