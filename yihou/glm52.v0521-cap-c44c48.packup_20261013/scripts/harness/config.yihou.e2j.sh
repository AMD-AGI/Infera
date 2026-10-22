#!/usr/bin/env bash
# Overlay (R050, round e2j): e2g control config + two DECODE-only levers:
#  (a) repro's DSA host-variant patch: graph_variants.py + eagle_utils.py mounts (decode only, via
#      DECODE_MOUNTS_ADD) + YIHOU_DSA_HOST_VARIANT=1 (skips the per-verify seq_lens.max().item() sync);
#  (b) --cuda-graph-bs-decode 1..16 (dense capture; default list was [1-8,10,12,14,16]).
# Optional E2J_DECODE_DEBUG_AGENT=1 adds the e2i decode-only debug-agent env (leader decides after e2i R1).
# Chain: E2J_BASE_CONFIG (default config.yihou.mc4340.sh -> MC4340_BASE_CONFIG).
_e2j_base="${E2J_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_e2j_base" ]] || { echo "config.yihou.e2j.sh: base not readable: $_e2j_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_e2j_base"
DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.decode-hostvariant.tsv"
DECODE_EXTRA_ENV+=" YIHOU_DSA_HOST_VARIANT=1"
DECODE_EXTRA_ARGS+=" --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
if [[ "${E2J_DECODE_DEBUG_AGENT:-0}" == 1 ]]; then
    DECODE_EXTRA_ENV+=" HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals"
fi
YIHOU_EXP="${YIHOU_EXP:-E}+e2j-hostvariant-bs1to16${E2J_DECODE_DEBUG_AGENT:+-dbg$E2J_DECODE_DEBUG_AGENT}"
unset _e2j_base
