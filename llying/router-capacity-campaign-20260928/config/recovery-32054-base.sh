#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1.sh
source "$TRACE_RUNTIME/config/selected-p.sh"
export ALLOCATION_JOB_ID=32054 PREFILL_ALLOCATION_JOB_ID=32054 DECODE_ALLOCATION_JOB_ID=32054
export PREFILL_NODE=smci355-ccs-aus-n04-33 PREFILL_IP=10.235.192.139
export DECODE_NODE=smci355-ccs-aus-n05-21 DECODE_IP=10.235.192.138
export CONTROL_NODE=$PREFILL_NODE TRACE_ENDPOINT=$PREFILL_IP:4317
export PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
export DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
export INFERA_SESSION_AFFINITY=prefill DECODE_RADIX=0
export DSA_PREFILL_BACKEND=tilelang DSA_DECODE_BACKEND=tilelang
export PERSIST_JIT_CACHE=1
export DECODE_DIAG_SOURCE=$TRACE_RUNTIME/docker/decode_prefix_diag.py
export RECOVERY_ALLOCATION_FILE=$TRACE_RUNTIME/recovery-allocation-32054.json
