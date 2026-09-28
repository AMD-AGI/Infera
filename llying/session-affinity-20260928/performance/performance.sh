#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/config/historical-base.sh
export TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928
export RUN_ID=session-affinity-31999-performance
export RUN=/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance
export CONTAINER_PREFIX=llying-session-31999-perf
export PREFILL_NODE=smci355-ccs-aus-n10-29
export PREFILL_IP=10.235.192.140
export DECODE_NODE=smci355-ccs-aus-n03-33
export DECODE_IP=10.235.192.56
export CONTROL_NODE=$PREFILL_NODE
export TOPOLOGY=/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/config/topology.tsv
export BENCH_DIR=/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/scripts/bench-harness
export REMOTE_BENCH_DIR=$BENCH_DIR
export ROUTER_BINARY_OVERRIDE=/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/artifacts/infera-router-session-v3
export ALLOCATION_JOB_ID=31999
export PREFILL_ALLOCATION_JOB_ID=31999
export DECODE_ALLOCATION_JOB_ID=31999
export SSH_OPTS='-F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts'
export PREFILL_HICACHE=1
export DECODE_HICACHE=0
export GUARD_MODE=completion
export INFERA_PD_PREFILL_GUARD_RELEASE=completion
export INFERA_R2_DECODE_DEMAND=off
export INFERA_R3_CACHE_TIERS=off
export INFERA_R4_PREFILL_WORK=off
export INFERA_SESSION_AFFINITY=prefill
export INFERA_SESSION_AFFINITY_TTL_SECS=3600
export KV_PREFILL_OVERLAP_WEIGHT=20
export KV_DECODE_OVERLAP_WEIGHT=2
export TRACE_ENDPOINT=$PREFILL_IP:4317
export PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
export DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
export RUST_LOG=info,infera_router::routing_experiments=warn
export SMOKE_ONLY=0

export DIAG_SOURCE_PREFILL=session-affinity-31999-performance
export DIAG_SOURCE_DECODE=session-affinity-31999-smoke
export BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/g0-guard-completion
export AGENTX_RUNTIME_VALIDATOR=$TRACE_RUNTIME/scripts/validate_and_pin_client.py
export AGENTX_DATASET_PINNER=$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
export PERFORMANCE_CONFIG_CONFIRMATION_REQUIRED=0

export DECODE_CONTAINER=llying-session-31999-decode-0

export AIPERF_PYTHON_INSTALL_DIR=/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/cache/python
