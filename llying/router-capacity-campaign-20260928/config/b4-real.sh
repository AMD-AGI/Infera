#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1.sh
source "$TRACE_RUNTIME/config/selected-p.sh"
source "$TRACE_RUNTIME/config/selected-d.sh"
export RUN_ID=campaign-b4-triton-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-b4-real
export PREVIOUS_RUN=$TRACE_RUNTIME/runs/campaign-b3-decode-affinity
export OLD_PREFIX=llying-campaign-b3
export TOPOLOGY=$TRACE_RUNTIME/config/b4-real.json
export DSA_PREFILL_BACKEND=triton DSA_DECODE_BACKEND=triton
export PERSIST_JIT_CACHE=1
export DECODE_DIAG_SOURCE=$TRACE_RUNTIME/docker/decode_prefix_diag.py
if [[ "$DECODE_RADIX" == 1 ]]; then
 export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
 export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
fi
export DECODE_SIMULATE_ACC_LEN=''
