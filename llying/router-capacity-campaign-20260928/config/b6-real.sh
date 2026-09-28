#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/recovery-32054-base.sh
source "$TRACE_RUNTIME/config/selected-p.sh"
source "$TRACE_RUNTIME/config/selected-d.sh"
source "$TRACE_RUNTIME/config/selected-backend.sh"
export RUN_ID=campaign-b6-2p8d8-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-b6-real
export PREVIOUS_RUN=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/runs/campaign-b4-triton
export OLD_PREFIX=llying-campaign-b4
export TOPOLOGY=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b6-real.json
export PREFILL_TP=8 PREFILL_DP=8 PREFILL_CHUNK_SIZE=32768
export PREFILL_MAX_RUNNING=256 PREFILL_GRAPH_MAX_BS=256
export DECODE_TP=8 DECODE_DP=8 DECODE_CHUNK_SIZE=32768
export DECODE_MAX_RUNNING=256 DECODE_GRAPH_MAX_BS=256
export PERSIST_JIT_CACHE=1
export DECODE_DIAG_SOURCE=$TRACE_RUNTIME/docker/decode_prefix_diag.py
if [[ "$DECODE_RADIX" == 1 ]]; then
 export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
 export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
fi
export DECODE_SIMULATE_ACC_LEN=''
