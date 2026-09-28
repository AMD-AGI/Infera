#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1.sh
source "$TRACE_RUNTIME/config/selected-p.sh"
export RUN_ID=campaign-b2-radix-real
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-b2-real
export OLD_PREFIX=llying-campaign-b1
export PREFILL_CONTAINER=llying-campaign-b1-prefill-0
export OLD_DECODE_CONTAINER=llying-campaign-b1-smoke-decode-0
export DECODE_CONTAINER=$CONTAINER_PREFIX-decode-0
export RESTART_PREFILL=0 RESTART_DECODE=1
export DECODE_RADIX=1
export DECODE_DIAG_SOURCE=$TRACE_RUNTIME/docker/decode_prefix_diag.py
export DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --disaggregation-decode-enable-radix-cache"
export DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
export DECODE_SIMULATE_ACC_LEN=''
export INFERA_SESSION_AFFINITY=both
export DIAG_SOURCE_PREFILL=campaign-b1-dynamo-p DIAG_SOURCE_DECODE=$RUN_ID
export PREVIOUS_RUN=$TRACE_RUNTIME/runs/campaign-b1-dynamo-p
