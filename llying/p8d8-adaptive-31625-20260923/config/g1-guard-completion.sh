#!/usr/bin/env bash
# Optimized configuration only, on job 31644 restart 1. No follow-up baseline.
source /perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/config/a0-guard-decode.sh
RUN_ID=g1-guard-completion
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
BASELINE_RUN="$TRACE_RUNTIME/runs/a0-guard-decode"
PREFILL_NODE=smci355-ccs-aus-n02-29
PREFILL_IP=10.235.192.61
DECODE_NODE=smci355-ccs-aus-n02-33
DECODE_IP=10.235.192.133
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
TOPOLOGY="$TRACE_RUNTIME/config/topology-31644-restart1.tsv"
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
GUARD_MODE=completion
DIAG_SOURCE_PREFILL="$RUN_ID"
DIAG_SOURCE_DECODE="$RUN_ID"
SERVICE_RUN_ID="$RUN_ID"
EXPECTED_PREFILL_TOKENS=3143424
EXPECTED_DECODE_TOKENS=3003264
EXPECTED_HOST_TOKENS=4715200
AGENTX_CLIENT_CONSTRAINTS="$TRACE_RUNTIME/config/client-constraints-a0.txt"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_and_pin_client.py"
export RUN_ID RUN BASELINE_RUN TRACE_RUNTIME CONFIG

SLURM_CONF="$TRACE_RUNTIME/config/slurm-client-dccs.conf"
PATH="/opt/slurm/bin:$PATH"
export SLURM_CONF PATH
