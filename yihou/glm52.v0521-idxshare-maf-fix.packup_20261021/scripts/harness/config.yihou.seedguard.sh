#!/usr/bin/env bash
# Overlay (R029): IndexShare seed guard + decode-only lean ROCm debug agent, on top of any chain
# (default: config.yihou.mc4340.sh, which itself sources MC4340_BASE_CONFIG).
#   SEEDGUARD_BASE_CONFIG   chain to source first (default config.yihou.mc4340.sh)
#   YIHOU_DSA_SEED_GUARD    off|count|mask (default count) -> DECODE env only
#   DECODE_DEBUG_AGENT      1 (default) | 0 -> lean debug agent on DECODE only (~1.00x, names the
#                           faulting kernel in the decode server log, also under graph replay)
# Mounts: v2-regbase + eagle_disaggregation / eagle_worker_v2 seedguard copies (repro, unit-tested).
_sg_base="${SEEDGUARD_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_sg_base" ]] || { echo "config.yihou.seedguard.sh: base not readable: $_sg_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_sg_base"

YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-regbase-seedguard.tsv"
: "${YIHOU_DSA_SEED_GUARD:=count}"
case "$YIHOU_DSA_SEED_GUARD" in off|count|mask) ;; *)
    echo "config.yihou.seedguard.sh: YIHOU_DSA_SEED_GUARD=$YIHOU_DSA_SEED_GUARD (want off|count|mask)" >&2
    return 2 2>/dev/null || exit 2 ;; esac
DECODE_EXTRA_ENV+=" YIHOU_DSA_SEED_GUARD=$YIHOU_DSA_SEED_GUARD"
if [[ "${DECODE_DEBUG_AGENT:-1}" == 1 ]]; then
    DECODE_EXTRA_ENV+=" HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals"
fi
YIHOU_EXP="${YIHOU_EXP:-E}+seedguard-$YIHOU_DSA_SEED_GUARD"
unset _sg_base
