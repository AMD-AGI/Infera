#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/config/run-common.sh
RUN_ID=r1r4-31719-performance-attempt2
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
CONTAINER_PREFIX=llying-r1r4-31719-performance-attempt2
AITER_JIT_CACHE_ROOT="/tmp/aiter-$RUN_ID"
PREFILL_HICACHE=1
SMOKE_ONLY=0
RUST_LOG=info,infera_router::routing_experiments=warn
export RUN RUN_ID SERVICE_RUN_ID TRACE_RUNTIME CONFIG
