#!/usr/bin/env bash
# AUS 1P1D P8D8, HiCache off. MODEL must name a populated path on both nodes.
AUS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?Set MODEL to the GLM-5.2-MXFP4 weights directory on both nodes}"
IMAGE="${IMAGE:-infera-sglang:v0519-llying-aus-0922-nextnfix-hicache}"
CONTROL_NODE=smci355-ccs-aus-n01-33
BUILDER_NODE=smci355-ccs-aus-n01-33
TOPOLOGY="$AUS_SCRIPT_DIR/topology.aus.tsv"
REMOTE_BENCH_DIR="$AUS_SCRIPT_DIR/bench-harness"
SSH_OPTS="${SSH_OPTS:--F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/agentx_known_hosts}"
CONTAINER_PREFIX=llying-aus-1p1d
PREFILL_HICACHE=0
DECODE_HICACHE=0
PREFILL_MAX_RUNNING=128
DECODE_MAX_RUNNING=128
PREFILL_GRAPH_MAX_BS=128
DECODE_GRAPH_MAX_BS=128
MC_GID_INDEX=1
MC_DISABLE_HIP_TRANSPORT=1
MOONCAKE_DISABLE_HIP_DMABUF=1
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}'
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
AGENTX_DURATION=1200
AGENTX_WARMUP_REQUESTS_PER_LANE=1
source "$AUS_SCRIPT_DIR/bench-harness/config.sh"
