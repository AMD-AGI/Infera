#!/usr/bin/env bash
# Fixed C80 -> C112 request-tracing contract. Do not use for C144.

TRACE_ROOT="/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c80-c112-tracing-20260922"
PACKUP_ROOT="/home/liyingli/bench_agentx/baseline/Infera/yihou/glm52.p8d8.agentx-sweep.packup_20260920"
BENCH_DIR="$PACKUP_ROOT/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"

IMAGE="infera-sglang:v0519-yihou-0917-nextnfix-hicache-reqtrace"
TRACE_IMAGE_ID_FILE="$TRACE_ROOT/image-id.txt"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-}"
if [[ -z "$EXPECTED_IMAGE_ID" && -s "$TRACE_IMAGE_ID_FILE" ]]; then
    EXPECTED_IMAGE_ID="$(<"$TRACE_IMAGE_ID_FILE")"
fi

PREFILL_NODE="crsuse2-m2m-138"
PREFILL_IP="10.245.157.237"
DECODE_NODE="crsuse2-m2m-136"
DECODE_IP="10.245.154.168"
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
CONTAINER_PREFIX="glm52-pd-c80-c112-trace"

PREFILL_GPU_DEVICES="0,1,2,3,4,5,6,7"
DECODE_GPU_DEVICES="0,1,2,3,4,5,6,7"
PREFILL_TP=8
PREFILL_DP=8
DECODE_TP=8
DECODE_DP=8
PREFILL_HICACHE=1
DECODE_HICACHE=0
PREFILL_HICACHE_RATIO=1.5
PREFILL_MAX_RUNNING=256
DECODE_MAX_RUNNING=256
PREFILL_GRAPH_MAX_BS=256
DECODE_GRAPH_MAX_BS=256

# C80 is first; the driver passes C112 to the second point without relaunching.
CONC=80
AGENTX_DURATION=3600
AGENTX_WARMUP_REQUESTS_PER_LANE=10
AGENTX_FAILED_REQUEST_THRESHOLD=0.10

PD_DP_RANK_AFFINITY=1
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}'
MC_TE_FILTERS="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
MC_GID_INDEX=1
MC_ENABLE_DEST_DEVICE_AFFINITY=1
MC_DISABLE_HIP_TRANSPORT=1
MOONCAKE_DISABLE_HIP_DMABUF=0

source "$PACKUP_ROOT/scripts/config.yihou.p8d8.sh"

# Built-in SGLang request-stage tracing plus human-auditable per-request duration
# lines. Async export keeps OTLP serialization off the scheduler hot path.
TRACE_ENDPOINT="${TRACE_ENDPOINT:-$PREFILL_IP:4317}"
TRACE_ARGS="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging"
TRACE_ENV="SGLANG_TRACE_ASYNC=1 SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD=64 SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500 SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=512"
PREFILL_EXTRA_ARGS="$TRACE_ARGS"
DECODE_EXTRA_ARGS="$TRACE_ARGS"
PREFILL_EXTRA_ENV="$TRACE_ENV"
DECODE_EXTRA_ENV="$TRACE_ENV"

_trace_require_eq() {
    local name="$1" expected="$2"
    if [[ "${!name-}" != "$expected" ]]; then
        echo "trace config drift: $name=${!name-<unset>} expected=$expected" >&2
        return 1
    fi
}

_trace_require_eq PREFILL_NODE "crsuse2-m2m-138"
_trace_require_eq DECODE_NODE "crsuse2-m2m-136"
_trace_require_eq PREFILL_HICACHE "1"
_trace_require_eq DECODE_HICACHE "0"
_trace_require_eq PREFILL_MAX_RUNNING "256"
_trace_require_eq DECODE_MAX_RUNNING "256"
_trace_require_eq AGENTX_WARMUP_REQUESTS_PER_LANE "10"
_trace_require_eq AGENTX_DURATION "3600"
_trace_require_eq PD_DP_RANK_AFFINITY "1"
_trace_require_eq JSON_MODEL_OVERRIDE_ARGS '{"index_share_for_mtp_iteration":false}'

export TRACE_ROOT PACKUP_ROOT BENCH_DIR REMOTE_BENCH_DIR
export PREFILL_NODE PREFILL_IP DECODE_NODE DECODE_IP
