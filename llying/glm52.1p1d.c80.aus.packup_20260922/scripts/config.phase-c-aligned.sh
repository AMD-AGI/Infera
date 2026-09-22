#!/usr/bin/env bash
# Align yihou C80's duration, warmup, scheduler caps and grouped-topk setting.
# Preserve the user's independent P/D rank routing choice and existing image.
ALIGNED_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ALIGNED_CONFIG_DIR/config.phase-b-hicache.sh"

PREFILL_MAX_RUNNING=256
DECODE_MAX_RUNNING=256
PREFILL_GRAPH_MAX_BS=256
DECODE_GRAPH_MAX_BS=256
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
AGENTX_DURATION=3600
AGENTX_WARMUP_REQUESTS_PER_LANE=10
PD_DP_RANK_AFFINITY=0
