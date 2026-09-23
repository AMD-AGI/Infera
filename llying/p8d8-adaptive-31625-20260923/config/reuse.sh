#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923/config/config.sh
TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923
RUN_ID="${CAMPAIGN_CASE_ID:?set a unique CAMPAIGN_CASE_ID}"
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923/runs/baseline4k-c80-31625
SERVICE_RUN_ID=baseline4k-c80-31625
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_chunk8k_runtime.py"
KV_PREFILL_OVERLAP_WEIGHT="${CAMPAIGN_P_OVERLAP:-20.0}"
export RUN_ID RUN TRACE_RUNTIME BASELINE_RUN SERVICE_RUN_ID
