#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/recovery-32054-base.sh
source "$TRACE_RUNTIME/config/selected-d.sh"
export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
export RUN_ID=campaign-rb-rebaseline-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-rb-real
export TOPOLOGY=$TRACE_RUNTIME/config/rb-real.json
export DECODE_SIMULATE_ACC_LEN=''
