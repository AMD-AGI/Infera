#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/recovery-base.sh
source "$TRACE_RUNTIME/config/selected-d.sh"
export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
export RUN_ID=campaign-r0-rebaseline-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-r0-real
export TOPOLOGY=$TRACE_RUNTIME/config/r0-real.json
export DECODE_SIMULATE_ACC_LEN=''
