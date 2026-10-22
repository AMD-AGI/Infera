#!/usr/bin/env bash
# Overlay (idxshare debug E1, 2026-10-20): find the kernel behind the decode "Memory access fault" with
# index_share_for_mtp_iteration=true. On top of the g4new chain (PR42553 + custom AR on), with:
#   - decode-only lean ROCm debug agent (validated R009: names the faulting kernel + PC under graph replay, ~1.00x)
#   - IndexShare seed guard in COUNT mode (R029: counts out-of-pool seed entries, does not change data)
#   - index_share forced TRUE (G4NEW_IDXSHARE=true)
# Mounts: mounts.yihou.v2-xferfix2-gtt-seedguard.tsv = v2-xferfix2-gtt + eagle_disaggregation/eagle_worker_v2
# seedguard copies (= image 9f5d8712 files + additions only, checked 2026-10-20).
#   DBG_BASE_CONFIG   chain to source (default config.yihou.g4new.sh)
#   YIHOU_DSA_SEED_GUARD  off|count|mask (default count)
#   DECODE_DEBUG_AGENT    1 (default) | 0
G4NEW_IDXSHARE=true
_dbg_base="${DBG_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.g4new.sh}"
[[ -r "$_dbg_base" ]] || { echo "config.yihou.dbgE1.sh: base not readable: $_dbg_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_dbg_base"
[[ "$INDEX_SHARE_FOR_MTP" == true ]] || { echo "config.yihou.dbgE1.sh: index_share must be true" >&2; return 2 2>/dev/null || exit 2; }
case "$YIHOU_MOUNTS" in
    "$WS/harness/mounts.yihou.v2-xferfix2-gtt.tsv") YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix2-gtt-seedguard.tsv" ;;
    *) echo "config.yihou.dbgE1.sh: unexpected YIHOU_MOUNTS=$YIHOU_MOUNTS" >&2; return 2 2>/dev/null || exit 2 ;;
esac
: "${YIHOU_DSA_SEED_GUARD:=count}"
case "$YIHOU_DSA_SEED_GUARD" in off|count|mask) ;; *)
    echo "config.yihou.dbgE1.sh: YIHOU_DSA_SEED_GUARD=$YIHOU_DSA_SEED_GUARD (want off|count|mask)" >&2
    return 2 2>/dev/null || exit 2 ;; esac
DECODE_EXTRA_ENV+=" YIHOU_DSA_SEED_GUARD=$YIHOU_DSA_SEED_GUARD"
if [[ "${DECODE_DEBUG_AGENT:-1}" == 1 ]]; then
    DECODE_EXTRA_ENV+=" HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals"
fi
YIHOU_EXP="${YIHOU_EXP:-E}+dbgE1-sg$YIHOU_DSA_SEED_GUARD-agent${DECODE_DEBUG_AGENT:-1}"
unset _dbg_base
