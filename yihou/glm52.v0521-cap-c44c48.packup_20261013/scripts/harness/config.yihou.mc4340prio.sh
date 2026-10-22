#!/usr/bin/env bash
# Overlay (task v0521cap, KV-send lever R043): run the mc4340 recipe on the mc4340prio image (102 only; = mc4340 +
# opt-in high-priority Mooncake HIP copy streams) with MC_HIP_STREAM_PRIORITY=high on BOTH legs (one image for both legs
# keeps the #4340 wire format consistent). Chain: PRIO_BASE_CONFIG (default config.yihou.mc4340.sh -> its base).
_prio_base="${PRIO_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
source "$_prio_base" || { return 2 2>/dev/null || exit 2; }
[[ "${YIHOU_NODE:-}" == crsuse2-m2m-102 ]] || { echo "config.yihou.mc4340prio.sh: mc4340prio image exists on 102 only" >&2; return 2 2>/dev/null || exit 2; }
IMAGE=infera-sglang:v0521-yihou-1010-mc4340prio
EXPECTED_IMAGE_ID=sha256:aacc9aeb46cebb411d29e99e6ce2afafe46349ca74f78d0d18d6e8b804624fcd
export IMAGE EXPECTED_IMAGE_ID
PREFILL_EXTRA_ENV+=" MC_HIP_STREAM_PRIORITY=${MC_PRIO:-high}"
DECODE_EXTRA_ENV+=" MC_HIP_STREAM_PRIORITY=${MC_PRIO:-high}"
YIHOU_EXP="${YIHOU_EXP:-E}+prio${MC_PRIO:-high}"
unset _prio_base
