#!/usr/bin/env bash
TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922
BASE_RUNTIME=/perf_apps/liyingli/bench_agentx/glm52-c80-20260922
MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
IMAGE=infera-sglang:aus-0922-reqtrace
source "$BASE_RUNTIME/scripts/config.phase-c-aligned.sh"
# Keep JSON stable when inherited config is sourced again by the harness.
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
RUN_ID="${RUN_ID:-main-20260922}"
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
INFERENCEX_DIR="$BASE_RUNTIME/scripts/bench-harness/.cache/InferenceX"
AGENTX_CACHE_DIR="$TRACE_RUNTIME/cache/agentx"
CONTAINER_PREFIX=llying-aus-trace
PREFILL_NODE=smci355-ccs-aus-n01-33
DECODE_NODE=smci355-ccs-aus-n02-21
PREFILL_IP=10.235.192.136
DECODE_IP=10.235.192.128
TRACE_ENDPOINT=10.235.192.136:4317
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging"
DECODE_EXTRA_ARGS="$PREFILL_EXTRA_ARGS"
TRACE_ENV="SGLANG_TRACE_LEVEL=1 SGLANG_TRACE_ASYNC=1 SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD=64 SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500 SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=512 AUS_DIAG_DIR=/aus-diag"
PREFILL_EXTRA_ENV="$TRACE_ENV AUS_DIAG_ROLE=prefill"
DECODE_EXTRA_ENV="$TRACE_ENV AUS_DIAG_ROLE=decode"
export RUN_ID RUN TRACE_RUNTIME

# Job 31625: reusable C80 4K baseline; overrides follow all baseline sources.
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/runs/main-20260922
TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923
RUN_ID=baseline4k-c80-31625
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
CONTAINER_PREFIX=llying-aus-4k-31625
PREFILL_NODE=smci355-ccs-aus-n03-33
DECODE_NODE=smci355-ccs-aus-n02-21
PREFILL_IP=10.235.192.56
DECODE_IP=10.235.192.128
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging"
DECODE_EXTRA_ARGS="$PREFILL_EXTRA_ARGS"
PREFILL_CHUNK_SIZE=32768
DECODE_CHUNK_SIZE=32768
export BASELINE_RUN RUN_ID RUN TRACE_RUNTIME
# Use the already-cached baseline dataset revision; do not refresh main.
AGENTX_DATASET_DOWNLOAD_OFFLINE=1
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_chunk8k_runtime.py"

ALLOCATION_JOB_ID=31625

# Match the completed 8K run on these same nodes; change only P chunk.
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923/runs/chunk8k-c80-31625
EXPECTED_PREFILL_CHUNK=4096

# Immutable A0 control on replacement allocation; both arms use one router binary.
ALLOCATION_JOB_ID=31644
TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923
RUN_ID=g0-guard-completion
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology-31644.tsv"
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-baseline4k-31625-20260923/runs/baseline4k-c80-31625
CONTAINER_PREFIX=llying-adaptive-31644
PREFILL_NODE=smci355-ccs-aus-n10-29
DECODE_NODE=smci355-ccs-aus-n02-21
PREFILL_IP=10.235.192.140
DECODE_IP=10.235.192.128
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
PREFILL_SEED=823508857
DECODE_SEED=19197414
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
GUARD_MODE=completion
ROUTER_BINARY_OVERRIDE="$TRACE_RUNTIME/artifacts/infera-router-guard"
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
EXPECTED_PREFILL_TOKENS=0
EXPECTED_DECODE_TOKENS=0
EXPECTED_HOST_TOKENS=0
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_chunk8k_runtime.py"
export RUN RUN_ID TRACE_RUNTIME BASELINE_RUN CONFIG

# Reuse the same engines; isolate this case with cursors and a new collector/router.
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode
DIAG_SOURCE_PREFILL=a0-guard-decode
DIAG_SOURCE_DECODE=a0-guard-decode
SERVICE_RUN_ID=a0-guard-decode
EXPECTED_PREFILL_TOKENS=3143424
EXPECTED_DECODE_TOKENS=3003264
EXPECTED_HOST_TOKENS=4715200
