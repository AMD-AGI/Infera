#!/usr/bin/env bash
# Overlay (R049): decode-only lean ROCm debug agent env, exactly the 3 vars config.yihou.seedguard.sh
# adds with DECODE_DEBUG_AGENT=1 (as in e2d) -- WITHOUT the seedguard mounts/env. Chain:
#   DBGAGENT_BASE_CONFIG (default config.yihou.mc4340.sh, which sources MC4340_BASE_CONFIG).
# audit_live.v2 detects the decode agent via HSA_TOOLS_LIB= in DECODE_EXTRA_ENV.
_da_base="${DBGAGENT_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_da_base" ]] || { echo "config.yihou.dbgagent.sh: base not readable: $_da_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_da_base"
DECODE_EXTRA_ENV+=" HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals"
YIHOU_EXP="${YIHOU_EXP:-E}+dbgagent-decode"
unset _da_base
