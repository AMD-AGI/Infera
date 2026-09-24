#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/config/resolved-base.sh
source /perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/config/nodes.sh
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
