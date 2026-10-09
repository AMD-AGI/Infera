#!/usr/bin/env bash
# Synthetic eight-expert target-only PD; source only inside the vendored harness.
WORKSPACE=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
BENCH_DIR="$WORKSPACE/scripts/bench-harness"
ORIGINAL_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
MODEL="$WORKSPACE/model"
IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache
SERVED_MODEL=glm5.2-synthetic-expert8-yihou
CONTROL_NODE=crsuse2-m2m-138
BUILDER_NODE=crsuse2-m2m-138
CONTAINER_PREFIX=glm52-pd-yihou-expert8
REMOTE_BENCH_DIR="$BENCH_DIR"
SSH_OPTS='-o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes'
TOPOLOGY="$WORKSPACE/scripts/topology.yihou.138-136.tsv"
PREFILL_GPU_DEVICES=0,1,2,3
DECODE_GPU_DEVICES=0,1,2,3
PREFILL_TP=4
PREFILL_DP=4
PREFILL_EP=4
PREFILL_DPA=1
DECODE_TP=4
DECODE_DP=4
DECODE_EP=4
DECODE_DPA=1
PREFILL_HICACHE=0
DECODE_HICACHE=0
DECODE_MTP=0
DECODE_SIMULATE_ACC_LEN=''
DSA_PREFILL_BACKEND=triton
DSA_DECODE_BACKEND=triton
DSA_TOPK_BACKEND=sgl-kernel
KV_CACHE_DTYPE=fp8_e4m3
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
PREFILL_MEM_FRACTION=0.85
DECODE_MEM_FRACTION=0.85
PREFILL_MAX_RUNNING=64
DECODE_MAX_RUNNING=64
PREFILL_GRAPH_MAX_BS=64
DECODE_GRAPH_MAX_BS=64
PREFILL_EXTRA_ARGS='--disable-shared-experts-fusion'
DECODE_EXTRA_ARGS='--disable-custom-all-reduce --disable-shared-experts-fusion'
PD_DP_RANK_AFFINITY=1
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3
MC_GID_INDEX=1
AITER_JIT_CACHE_ROOT="$WORKSPACE/cache/${role:-control}/aiter"
SGLANG_MOUNT_MANIFEST="$WORKSPACE/patches/mounts.yihou.tsv"
: "${READY_TIMEOUT:=3600}"
: "${ROUTER_READY_TIMEOUT:=600}"
source "$BENCH_DIR/config.sh"
# Reapply after the inherited default's unquoted-brace expansion.
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
