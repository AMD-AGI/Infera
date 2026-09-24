#!/usr/bin/env bash
source /perf_apps/liyingli/bench_agentx/router-r234-31699-20260924/config/baseline-template.sh
TRACE_RUNTIME=/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924
ALLOCATION_JOB_ID=31699
PREFILL_NODE=smci355-ccs-aus-n02-29
DECODE_NODE=smci355-ccs-aus-n06-25
PREFILL_IP=10.235.192.61
DECODE_IP=10.235.192.59
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
TOPOLOGY="$TRACE_RUNTIME/config/topology.tsv"
CONTAINER_PREFIX=llying-adaptive-31699
SSH_OPTS="-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$TRACE_RUNTIME/config/known_hosts"
SLURM_CONF="$TRACE_RUNTIME/config/slurm-client-dccs.conf"
PATH="/opt/slurm/bin:$PATH"
PREFILL_SEED=434370234
DECODE_SEED=30296820
PREFILL_CHUNK_SIZE=65536
DECODE_CHUNK_SIZE=32768
EXPECTED_PREFILL_CHUNK=8192
PREFILL_MEM_FRACTION=0.85
DECODE_MEM_FRACTION=0.85
PREFILL_HICACHE_RATIO=1.5
PREFILL_MAX_TOTAL_TOKENS=0
EXPECTED_PREFILL_TOKENS=0
EXPECTED_DECODE_TOKENS=0
EXPECTED_HOST_TOKENS=0
TRACE_ENDPOINT="$PREFILL_IP:4317"
PREFILL_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging --random-seed $DECODE_SEED"
ROUTER_BINARY_OVERRIDE="$TRACE_RUNTIME/artifacts/infera-router-r234"
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_and_pin_client.py"
AGENTX_CLIENT_CONSTRAINTS="$TRACE_RUNTIME/config/client-constraints.txt"
READY_TIMEOUT=7200
GPU_IDLE_TIMEOUT=7200
GUARD_MODE=decode
INFERA_R2_DECODE_DEMAND=shadow
INFERA_R3_CACHE_TIERS=shadow
INFERA_R3_HOST_WEIGHT=0.5
INFERA_R4_PREFILL_WORK=off
export SLURM_CONF PATH
