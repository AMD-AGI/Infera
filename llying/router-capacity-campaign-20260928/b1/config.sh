#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1-smoke.sh
export RUN_ID=campaign-b1-dynamo-p
export RUN=$TRACE_RUNTIME/runs/$RUN_ID
export CONTAINER_PREFIX=llying-campaign-b1
export PREFILL_HICACHE=1
export DECODE_CONTAINER=llying-campaign-b1-smoke-decode-0
export DIAG_SOURCE_PREFILL=$RUN_ID DIAG_SOURCE_DECODE=campaign-b1-smoke
export RUST_LOG=info,infera_router::routing_experiments=warn
