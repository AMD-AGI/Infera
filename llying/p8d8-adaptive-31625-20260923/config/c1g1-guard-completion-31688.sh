#!/usr/bin/env bash
# Keep C1's engines/capacity; measure only the router guard release change.
source /perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/config/c1-fixed-host-31688.sh
RUN_ID=c1g1-guard-completion-31688
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BASELINE_RUN="$TRACE_RUNTIME/runs/c1-fixed-host-31688"
GUARD_MODE=completion
DIAG_SOURCE_PREFILL=c1-fixed-host-31688
DIAG_SOURCE_DECODE=c1-fixed-host-31688
SERVICE_RUN_ID=c1-fixed-host-31688
export RUN_ID RUN BASELINE_RUN GUARD_MODE DIAG_SOURCE_PREFILL DIAG_SOURCE_DECODE SERVICE_RUN_ID
