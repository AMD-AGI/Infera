#!/usr/bin/env bash
# Fixed contract for the cross-rank Mooncake RCA and its final acceptance run.
# This file intentionally sets every load-bearing delta before sourcing the
# packup P8D8 config, then verifies that no inherited default changed it.
# The 2026-09-21 reproduction ran with Prefill on crsuse2-m2m-138; since
# 2026-09-28 the pair is 137->136. A fix run sets RCA_IMAGE/RCA_IMAGE_ID in a
# wrapper config that sources this file; nothing else may differ.

RCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKUP_ROOT="/home/liyingli/bench_agentx/baseline/Infera/yihou/glm52.p8d8.agentx-sweep.packup_20260920"
BENCH_DIR="$PACKUP_ROOT/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"

IMAGE="${RCA_IMAGE:-infera-sglang:v0519-yihou-0917-nextnfix-hicache}"
EXPECTED_IMAGE_ID="${RCA_IMAGE_ID:-sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35}"
_rca_image="$IMAGE"
PREFILL_NODE="crsuse2-m2m-137"
PREFILL_IP="10.245.153.247"
DECODE_NODE="crsuse2-m2m-136"
DECODE_IP="10.245.154.168"
CONTROL_NODE="$PREFILL_NODE"
BUILDER_NODE="$PREFILL_NODE"
CONTAINER_PREFIX="glm52-pd-crossrank-rca"

PREFILL_GPU_DEVICES="0,1,2,3,4,5,6,7"
DECODE_GPU_DEVICES="0,1,2,3,4,5,6,7"
PREFILL_TP=8
PREFILL_DP=8
DECODE_TP=8
DECODE_DP=8
PREFILL_HICACHE=0
DECODE_HICACHE=0
PREFILL_MAX_RUNNING=256
DECODE_MAX_RUNNING=256
PREFILL_GRAPH_MAX_BS=256
DECODE_GRAPH_MAX_BS=256

CONC=80
AGENTX_DURATION=1200
AGENTX_WARMUP_REQUESTS_PER_LANE=1
AGENTX_FAILED_REQUEST_THRESHOLD=0.10

PD_DP_RANK_AFFINITY=0
RDMA_DEVICE="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
MC_TE_FILTERS="$RDMA_DEVICE"
MC_GID_INDEX=1
MC_ENABLE_DEST_DEVICE_AFFINITY=1
MC_DISABLE_HIP_TRANSPORT=1
MOONCAKE_DISABLE_HIP_DMABUF=0
SGLANG_ENABLE_FAILED_SESSION_PROBE=1
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=30

# Keep the packup's stable compute-side controls and simulated acceptance. The
# source performs the post-source IndexShare assignment in the required order.
source "$PACKUP_ROOT/scripts/config.yihou.p8d8.sh"

_rca_require_eq() {
    local name="$1" expected="$2"
    if [[ "${!name-}" != "$expected" ]]; then
        echo "RCA config drift: $name=${!name-<unset>} expected=$expected" >&2
        return 1
    fi
}

_rca_require_eq IMAGE "$_rca_image"
_rca_require_eq PREFILL_NODE "crsuse2-m2m-137"
_rca_require_eq DECODE_NODE "crsuse2-m2m-136"
_rca_require_eq CONTROL_NODE "crsuse2-m2m-137"
_rca_require_eq CONTAINER_PREFIX "glm52-pd-crossrank-rca"
_rca_require_eq PREFILL_HICACHE "0"
_rca_require_eq DECODE_HICACHE "0"
_rca_require_eq PD_DP_RANK_AFFINITY "0"
_rca_require_eq RDMA_DEVICE "ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
_rca_require_eq MC_TE_FILTERS "$RDMA_DEVICE"
_rca_require_eq AGENTX_WARMUP_REQUESTS_PER_LANE "1"
_rca_require_eq AGENTX_DURATION "1200"
_rca_require_eq DECODE_SIMULATE_ACC_LEN "3.61"
_rca_require_eq JSON_MODEL_OVERRIDE_ARGS '{"index_share_for_mtp_iteration":false}'

if [[ "$RDMA_DEVICE" == *"{"* ]]; then
    echo "RCA config drift: per-GPU RDMA JSON is forbidden" >&2
    return 1 2>/dev/null || exit 1
fi

export RCA_ROOT PACKUP_ROOT BENCH_DIR REMOTE_BENCH_DIR EXPECTED_IMAGE_ID
export PREFILL_NODE PREFILL_IP DECODE_NODE DECODE_IP
# Python helpers (scripts/rca_nodes.py) follow the same pair and image.
export RCA_PREFILL_NODE="$PREFILL_NODE" RCA_DECODE_NODE="$DECODE_NODE"
export RCA_IMAGE="$IMAGE" RCA_IMAGE_ID="$EXPECTED_IMAGE_ID"
