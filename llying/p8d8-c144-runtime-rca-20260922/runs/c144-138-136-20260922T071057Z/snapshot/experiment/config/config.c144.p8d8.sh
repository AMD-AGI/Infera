#!/usr/bin/env bash
# Fixed contract for the P8D8 C144 runtime RCA.
#
# This reproduces the 2026-09-20 HiCache-on sweep configuration except for the
# unavailable prefill host: crsuse2-m2m-138 replaces crsuse2-m2m-137. Decode
# remains on crsuse2-m2m-136.

RCA_ROOT="/home/liyingli/bench_agentx/baseline/Infera/bench/glm5p2_pd/results/p8d8-c144-runtime-rca-20260922"
PACKUP_ROOT="/home/liyingli/bench_agentx/baseline/Infera/yihou/glm52.p8d8.agentx-sweep.packup_20260920"
BENCH_DIR="$PACKUP_ROOT/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"

IMAGE="infera-sglang:v0519-yihou-0917-nextnfix-hicache"
EXPECTED_IMAGE_ID="sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"
PREFILL_NODE="crsuse2-m2m-138"
PREFILL_IP="10.245.157.237"
DECODE_NODE="crsuse2-m2m-136"
DECODE_IP="10.245.154.168"
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
CONTAINER_PREFIX="glm52-pd-c144-runtime-rca"

PREFILL_GPU_DEVICES="0,1,2,3,4,5,6,7"
DECODE_GPU_DEVICES="0,1,2,3,4,5,6,7"
PREFILL_TP=8
PREFILL_DP=8
DECODE_TP=8
DECODE_DP=8

# Preserve the sweep-of-record cache hierarchy.
PREFILL_HICACHE=1
DECODE_HICACHE=0
PREFILL_HICACHE_RATIO=1.5
PREFILL_HICACHE_WRITE_POLICY=write_through
PREFILL_HICACHE_IO_BACKEND=kernel
PREFILL_HICACHE_MEM_LAYOUT=page_first

# Global budgets. With attention DP=8 the request pool is at most 32 requests
# per rank. C144's historical Effective Concurrency max was 248, below the
# global 256 ceiling; the sampler will detect a hot rank reaching 32.
PREFILL_MAX_RUNNING=256
DECODE_MAX_RUNNING=256
PREFILL_GRAPH_MAX_BS=256
DECODE_GRAPH_MAX_BS=256

CONC=144
AGENTX_DURATION=3600
AGENTX_WARMUP_REQUESTS_PER_LANE=10
AGENTX_FAILED_REQUEST_THRESHOLD=0.10

# Preserve same-rank routing and GPU_n -> ionic_n pairing from the original
# P8D8 sweep. MC_TE_FILTERS must remain a plain comma list, not this JSON map.
PD_DP_RANK_AFFINITY=1
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}'
MC_TE_FILTERS="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
MC_GID_INDEX=1
MC_ENABLE_DEST_DEVICE_AFFINITY=1
MC_DISABLE_HIP_TRANSPORT=1
MOONCAKE_DISABLE_HIP_DMABUF=0
SGLANG_ENABLE_FAILED_SESSION_PROBE=1
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=30

# The packup config applies stable compute controls, simulated acceptance 3.61,
# and the post-source IndexShare=false workaround in the required order.
source "$PACKUP_ROOT/scripts/config.yihou.p8d8.sh"

_rca_require_eq() {
    local name="$1" expected="$2"
    if [[ "${!name-}" != "$expected" ]]; then
        echo "RCA config drift: $name=${!name-<unset>} expected=$expected" >&2
        return 1
    fi
}

_rca_require_eq IMAGE "infera-sglang:v0519-yihou-0917-nextnfix-hicache"
_rca_require_eq PREFILL_NODE "crsuse2-m2m-138"
_rca_require_eq DECODE_NODE "crsuse2-m2m-136"
_rca_require_eq CONTROL_NODE "crsuse2-m2m-138"
_rca_require_eq CONTAINER_PREFIX "glm52-pd-c144-runtime-rca"
_rca_require_eq CONC "144"
_rca_require_eq PREFILL_HICACHE "1"
_rca_require_eq DECODE_HICACHE "0"
_rca_require_eq PREFILL_MAX_RUNNING "256"
_rca_require_eq DECODE_MAX_RUNNING "256"
_rca_require_eq PREFILL_GRAPH_MAX_BS "256"
_rca_require_eq DECODE_GRAPH_MAX_BS "256"
_rca_require_eq PD_DP_RANK_AFFINITY "1"
_rca_require_eq AGENTX_WARMUP_REQUESTS_PER_LANE "10"
_rca_require_eq AGENTX_DURATION "3600"
_rca_require_eq DECODE_SIMULATE_ACC_LEN "3.61"
_rca_require_eq JSON_MODEL_OVERRIDE_ARGS '{"index_share_for_mtp_iteration":false}'

export RCA_ROOT PACKUP_ROOT BENCH_DIR REMOTE_BENCH_DIR EXPECTED_IMAGE_ID
export PREFILL_NODE PREFILL_IP DECODE_NODE DECODE_IP
