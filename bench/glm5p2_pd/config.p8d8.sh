#!/usr/bin/env bash
# Fixed P8+DPA / D8+DPA profile for the 137/138 AgentX campaign.
# Per-point max-running and graph limits are still passed explicitly.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

PREFILL_GPU_DEVICES="0,1,2,3,4,5,6,7"
PREFILL_TP=8
PREFILL_EP=1
PREFILL_DP=8
PREFILL_DPA=1

DECODE_GPU_DEVICES="0,1,2,3,4,5,6,7"
DECODE_TP=8
DECODE_EP=1
DECODE_DP=8
DECODE_DPA=1

PREFILL_MAX_RUNNING=32
PREFILL_GRAPH_MAX_BS=32
DECODE_MAX_RUNNING=32
DECODE_GRAPH_MAX_BS=32

CONTAINER_PREFIX="glm52-pd-p8dpa-d8dpa-v518"
AGENTX_WARMUP_REQUESTS_PER_LANE=10
