#!/usr/bin/env bash
# Retry after pre-load client pin validation failure; preserve the failed run.
source /perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/config/c1g1-guard-completion-31688.sh
RUN_ID=c1g1r-guard-completion-31688
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
export RUN_ID RUN
