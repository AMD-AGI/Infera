#!/usr/bin/env bash
# Bind only after the new allocation and tested release binary are known.
source /perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924/config/baseline-template.sh
PREFILL_ALLOCATION_JOB_ID="${R2_PREFILL_JOB_ID:?Prefill allocation required}"
DECODE_ALLOCATION_JOB_ID="${R2_DECODE_JOB_ID:?Decode allocation required}"
# Compatibility label only; allocation checks must validate both role-specific jobs.
ALLOCATION_JOB_ID="$PREFILL_ALLOCATION_JOB_ID"
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
CONTAINER_PREFIX="llying-r2-$PREFILL_ALLOCATION_JOB_ID-$DECODE_ALLOCATION_JOB_ID"
BASELINE_RUN=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode
ROUTER_BINARY_OVERRIDE="${R2_ROUTER_BINARY:?tested release binary required}"
SSH_OPTS="-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$TRACE_RUNTIME/config/known_hosts"

# Freeze the reviewed model/scheduler settings after loading the historical template.
IMAGE=infera-sglang:aus-0922-reqtrace
MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
SERVED_MODEL=glm5.2-mxfp4
PREFILL_GPU_DEVICES=0,1,2,3,4,5,6,7
DECODE_GPU_DEVICES=0,1,2,3,4,5,6,7
PREFILL_TP=8
DECODE_TP=8
PREFILL_DP=8
DECODE_DP=8
PREFILL_EP=1
DECODE_EP=1
PREFILL_DPA=1
DECODE_DPA=1
PREFILL_MAX_RUNNING=256
DECODE_MAX_RUNNING=256
PREFILL_GRAPH_MAX_BS=256
DECODE_GRAPH_MAX_BS=256
KV_CACHE_DTYPE=fp8_e4m3
PREFILL_HICACHE_WRITE_POLICY=write_through
PREFILL_HICACHE_IO_BACKEND=kernel
PREFILL_HICACHE_MEM_LAYOUT=page_first
DECODE_MTP=1
DECODE_SPEC_STEPS=5
DECODE_SPEC_TOPK=1
DECODE_SPEC_DRAFT_TOKENS=6
DECODE_SIMULATE_ACC_LEN=3.61
KV_PREFILL_OVERLAP_WEIGHT=20.0
KV_DECODE_OVERLAP_WEIGHT=2.0
SGLANG_OPT_USE_TOPK_V2=false
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
AITER_ALLREDUCE_FUSION=1
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
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

SLURM_CONF="$TRACE_RUNTIME/config/slurm-client-dccs.conf"
PATH="/opt/slurm/bin:$PATH"
export SLURM_CONF PATH
