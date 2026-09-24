#!/usr/bin/env bash
# Bind only after the new allocation and tested release binary are known.
source /perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/config/a0-guard-decode.sh
ALLOCATION_JOB_ID="${R2_ALLOCATION_JOB_ID:?allocation required}"
TRACE_RUNTIME="${R2_RUNTIME:?isolated runtime required}"
RUN_ID="${R2_RUN_TAG:?unique run tag required}"
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
PREFILL_NODE="${R2_PREFILL_NODE:?assigned node required}"
DECODE_NODE="${R2_DECODE_NODE:?assigned node required}"
PREFILL_IP="${R2_PREFILL_IP:?assigned address required}"
DECODE_IP="${R2_DECODE_IP:?assigned address required}"
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
CONTAINER_PREFIX="llying-r2-$ALLOCATION_JOB_ID"
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode
ROUTER_BINARY_OVERRIDE="${R2_ROUTER_BINARY:?tested release binary required}"
SSH_OPTS="-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$TRACE_RUNTIME/config/known_hosts"

PREFILL_CHUNK_SIZE=32768
DECODE_CHUNK_SIZE=32768
EXPECTED_PREFILL_CHUNK=4096
PREFILL_SEED=823508857
DECODE_SEED=19197414
PREFILL_MEM_FRACTION=0.85
DECODE_MEM_FRACTION=0.85
PREFILL_MAX_TOTAL_TOKENS=0
PREFILL_HICACHE_RATIO=1.5
DECODE_HICACHE=0
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"

GUARD_MODE=decode
INFERA_PD_PREFILL_GUARD_RELEASE=decode
INFERA_R2_DECODE_DEMAND=on
INFERA_R3_CACHE_TIERS=off
INFERA_R3_HOST_WEIGHT=0
INFERA_R4_PREFILL_WORK=off
PD_DP_RANK_AFFINITY=0
CONC=80
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
AITER_JIT_CACHE_ROOT="/tmp/aiter-$RUN_ID"
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_and_pin_client.py"
AGENTX_CLIENT_CONSTRAINTS="$TRACE_RUNTIME/config/client-constraints.txt"
export RUN RUN_ID TRACE_RUNTIME BASELINE_RUN CONFIG
