#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b2.sh
export RUN_ID=campaign-b3-decode-affinity
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-b3
export OLD_PREFIX=llying-campaign-b2
export DECODE_CONTAINER=llying-campaign-b2-decode-0
export RESTART_PREFILL=0 RESTART_DECODE=0
export INFERA_SESSION_AFFINITY=both
export DIAG_SOURCE_DECODE=campaign-b2-radix
export PREVIOUS_RUN=$TRACE_RUNTIME/runs/campaign-b2-radix
