#!/usr/bin/env bash
# aus B4 ("组合切换Triton") on crsuse2. Values are B4's resolved environment
# (router-capacity-campaign-20260928/config/b4.sh and the files it sources);
# only the lines marked "crsuse2" differ.
# Node pair: B4_PREFILL_NODE / B4_DECODE_NODE (default 136 -> 138).

B4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_b4_ip() {
    case "$1" in
        crsuse2-m2m-136) echo 10.245.154.168 ;;
        crsuse2-m2m-137) echo 10.245.153.247 ;;
        crsuse2-m2m-138) echo 10.245.157.237 ;;
    esac
}

# ---- crsuse2 bindings ----
PREFILL_NODE="${B4_PREFILL_NODE:-crsuse2-m2m-136}"
DECODE_NODE="${B4_DECODE_NODE:-crsuse2-m2m-138}"
PREFILL_IP="$(_b4_ip "$PREFILL_NODE")"
DECODE_IP="$(_b4_ip "$DECODE_NODE")"
CONTROL_NODE="$PREFILL_NODE"
TRACE_RUNTIME="$B4_ROOT/runtime"
BENCH_DIR="$TRACE_RUNTIME/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
# run_b4.sh exports B4_RUN_ID so every process that sources this file agrees.
RUN_ID="${B4_RUN_ID:-b4-crsuse2-${PREFILL_NODE##*-}-${DECODE_NODE##*-}-$(date -u +%Y%m%dT%H%MZ)}"
RUN="$TRACE_RUNTIME/runs/$RUN_ID"
TOPOLOGY="$RUN/topology.json"
# transition_two_node.stop() only stops containers under llying-campaign-.
CONTAINER_PREFIX=llying-campaign-b4crs
ALLOCATION_JOB_ID=none PREFILL_ALLOCATION_JOB_ID=none DECODE_ALLOCATION_JOB_ID=none
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
IMAGE=infera-sglang:aus-campaign-radix-20260928-mcdestpin-ibto18
EXPECTED_IMAGE_ID=sha256:e13b584341d6fa0a2c3cae2f1894f8f9f805d50e920506e5d44133b42ad34d0a
ROUTER_BINARY_OVERRIDE="$B4_ROOT/artifacts/infera-router-19a6c1d2"
# Two clusters' NIC drivers differ; crsuse2 runs with HIP DMABUF on.
MOONCAKE_DISABLE_HIP_DMABUF=0
AGENTX_CACHE_DIR="$TRACE_RUNTIME/cache/agentx"
AGENTX_CLIENT_CONSTRAINTS="$TRACE_RUNTIME/config/client-constraints.txt"
AGENTX_DATASET_PINNER="$TRACE_RUNTIME/scripts/pin_chunk8k_dataset.py"
AGENTX_RUNTIME_VALIDATOR="$TRACE_RUNTIME/scripts/validate_and_pin_client.py"
# B4's runtime.env with only its aus paths rewritten (build/make_reference.sh):
# the validator fails the run if any other client setting differs from B4.
BASELINE_RUN="$TRACE_RUNTIME/reference/b4"
AIPERF_PYTHON_INSTALL_DIR="$TRACE_RUNTIME/cache/python"
INFERENCEX_DIR="$TRACE_RUNTIME/cache/InferenceX"
DECODE_DIAG_SOURCE="$TRACE_RUNTIME/docker/decode_prefix_diag.py"
# B4's options plus LogLevel=ERROR: transition_two_node.remote() merges stderr
# into the output it parses, and a first-contact host-key notice would corrupt it.
SSH_OPTS='-F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts -o LogLevel=ERROR'

# ---- B4 recipe ----
CONC=80 DURATION=3600 AGENTX_DURATION=3600
# Cached Weka snapshot 23f152f6, file-for-file identical to B4's manifest.
AGENTX_DATASET_DOWNLOAD_OFFLINE=1
AGENTX_WARMUP_REQUESTS_PER_LANE=10
AGENTX_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
INFERENCEX_REPOSITORY=https://github.com/SemiAnalysisAI/InferenceX.git
INFERENCEX_REF=918524ff94045b3f091115f1051c22a8588edf2b
SERVED_MODEL=glm5.2-mxfp4
KV_P2P_TRANSFER=mooncake KV_CACHE_DTYPE=fp8_e4m3
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
DSA_PREFILL_BACKEND=triton DSA_DECODE_BACKEND=triton
AITER_ALLREDUCE_FUSION=1
ENABLE_KV_AWARE=1
ENGINE_PORT_BASE=29001 BOOTSTRAP_PORT_BASE=28998 KV_EVENT_PORT_BASE=25557 SNAPSHOT_PORT_BASE=28801
ETCD_PORT=22379 ETCD_PEER_PORT=22380 ROUTER_PORT=28000
READY_TIMEOUT=3600 ROUTER_READY_TIMEOUT=300 HEALTH_INTERVAL=10
# Value placement-ports.sh exported for the B4 run (launch_placement.py asserts it).
INFERA_NODEPORT_RANGE=25000-32767
SGLANG_ENABLE_FAILED_SESSION_PROBE=1 SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=30
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 SGLANG_OPT_USE_TOPK_V2=false
SGLANG_TIMEOUT_KEEP_ALIVE=900
MC_GID_INDEX=1 MC_DISABLE_HIP_TRANSPORT=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}'
RDMAV_FORK_SAFE=1 NCCL_IB_DISABLE=1
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so HOST_RDMA_MOUNT=/host-libionic/libionic.so
PERSIST_JIT_CACHE=1
PD_DP_RANK_AFFINITY=0

PREFILL_TP=8 PREFILL_DP=8 PREFILL_DPA=1 PREFILL_EP=1
PREFILL_GPU_DEVICES=0,1,2,3,4,5,6,7
PREFILL_CHUNK_SIZE=32768 PREFILL_MAX_RUNNING=256 PREFILL_GRAPH_MAX_BS=256
PREFILL_MEM_FRACTION=0.85 PREFILL_MAX_TOTAL_TOKENS=0
PREFILL_HICACHE=1 PREFILL_HICACHE_RATIO=1.5 PREFILL_HICACHE_WRITE_POLICY=write_through
PREFILL_HICACHE_IO_BACKEND=kernel PREFILL_HICACHE_MEM_LAYOUT=page_first
PREFILL_HSA_NO_SCRATCH_RECLAIM=0
PREFILL_SEED=823508857

DECODE_TP=8 DECODE_DP=8 DECODE_DPA=1 DECODE_EP=1
DECODE_GPU_DEVICES=0,1,2,3,4,5,6,7
DECODE_CHUNK_SIZE=32768 DECODE_MAX_RUNNING=256 DECODE_GRAPH_MAX_BS=256
DECODE_MEM_FRACTION=0.85
DECODE_HICACHE=0 DECODE_HICACHE_RATIO=1.5 DECODE_HICACHE_WRITE_POLICY=write_through
DECODE_HICACHE_IO_BACKEND=kernel DECODE_HICACHE_MEM_LAYOUT=page_first
DECODE_HSA_NO_SCRATCH_RECLAIM=1
DECODE_MTP=1 DECODE_SPEC_STEPS=5 DECODE_SPEC_TOPK=1 DECODE_SPEC_DRAFT_TOKENS=6
DECODE_SEED=19197414
DECODE_RADIX=1
# Real acceptance for the answer gate; run_b4.sh relaunches decode at 3.61.
DECODE_SIMULATE_ACC_LEN="${B4_DECODE_SIMULATE_ACC_LEN-3.61}"

# Router (start_performance_router.py reads these; GUARD_MODE is its record).
GUARD_MODE=completion INFERA_PD_PREFILL_GUARD_RELEASE=completion
INFERA_P_DYNAMO_SCORE=off INFERA_R2_DECODE_DEMAND=off INFERA_R3_CACHE_TIERS=off
INFERA_R3_HOST_WEIGHT=0 INFERA_R4_PREFILL_WORK=off
INFERA_SESSION_AFFINITY=both INFERA_SESSION_AFFINITY_TTL_SECS=3600
KV_PREFILL_OVERLAP_WEIGHT=20 KV_DECODE_OVERLAP_WEIGHT=2
RUST_LOG=info,infera_router::routing_experiments=warn

# Capacity reference (tokens per rank) for the post-launch check.
EXPECTED_PREFILL_CHUNK=4096
REFERENCE_PREFILL_TOKENS=3143424 REFERENCE_HOST_TOKENS=4715200 REFERENCE_DECODE_TOKENS=3003264
MAX_AUTOMATIC_CAPACITY_RELATIVE_DRIFT=0.001

# Tracing and diagnostics, as in B4.
TRACE_ENDPOINT="$PREFILL_IP:4317"
TRACE_ENV='SGLANG_TRACE_LEVEL=1 SGLANG_TRACE_ASYNC=1 SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD=64 SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500 SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=512 AUS_DIAG_DIR=/aus-diag'
PREFILL_EXTRA_ENV="$TRACE_ENV AUS_DIAG_ROLE=prefill"
DECODE_EXTRA_ENV="$TRACE_ENV AUS_DIAG_ROLE=decode SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1"
_b4_trace="--enable-trace --trace-modules request,mooncake --otlp-traces-endpoint $TRACE_ENDPOINT --enable-request-time-stats-logging"
PREFILL_EXTRA_ARGS="$_b4_trace --random-seed $PREFILL_SEED"
DECODE_EXTRA_ARGS="$_b4_trace --random-seed $DECODE_SEED --disaggregation-decode-enable-radix-cache"

if [[ -z "$PREFILL_IP" || -z "$DECODE_IP" || "$PREFILL_NODE" == "$DECODE_NODE" ]]; then
    echo "b4 config: invalid node pair $PREFILL_NODE -> $DECODE_NODE" >&2
    return 1 2>/dev/null || exit 1
fi
