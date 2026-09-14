#!/usr/bin/env bash
# Site configuration for the two-node GLM-5.2 1P1D experiment.
# Usage: export PREFILL_NODE=... DECODE_NODE=...; source ./config.sh
# Override any value from the environment; no other script needs editing.

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PREFILL_NODE:-}" || -z "${DECODE_NODE:-}" ]]; then
    echo "[config] run inventory_nodes.sh, then export PREFILL_NODE and DECODE_NODE" >&2
    return 64 2>/dev/null || exit 64
fi
if [[ "$PREFILL_NODE" == "$DECODE_NODE" ]]; then
    echo "[config] prefill and decode must use different nodes: $PREFILL_NODE" >&2
    return 64 2>/dev/null || exit 64
fi
export PREFILL_NODE DECODE_NODE
export DATA_NET="${DATA_NET:-10.245.}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no}"
if [[ "$PREFILL_NODE" < "$DECODE_NODE" ]]; then
    BENCH_NODE_PAIR="${PREFILL_NODE}_${DECODE_NODE}"
else
    BENCH_NODE_PAIR="${DECODE_NODE}_${PREFILL_NODE}"
fi
export BENCH_LOCK_FILE="${BENCH_LOCK_FILE:-${TMPDIR:-/tmp}/glm5p2_1p1d_${BENCH_NODE_PAIR}.lock}"

acquire_bench_lock() {
    [[ "${GLM52_BENCH_LOCK_HELD:-0}" == "1" ]] && return 0
    command -v flock >/dev/null 2>&1 || {
        echo "[config] flock is required for lifecycle serialization" >&2
        return 69
    }
    exec {GLM52_BENCH_LOCK_FD}>"$BENCH_LOCK_FILE"
    flock -n "$GLM52_BENCH_LOCK_FD" || {
        echo "[config] another run owns $PREFILL_NODE/$DECODE_NODE" >&2
        return 75
    }
    export GLM52_BENCH_LOCK_FD GLM52_BENCH_LOCK_HELD=1
}

node_ip() {
    local node="$1"
    ssh $SSH_OPTS "$node" "hostname -I" \
        | tr ' ' '\n' \
        | awk -v prefix="$DATA_NET" 'index($0, prefix) == 1 { print; exit }'
}

export PREFILL_IP="${PREFILL_IP:-$(node_ip "$PREFILL_NODE")}"
export DECODE_IP="${DECODE_IP:-$(node_ip "$DECODE_NODE")}"
[[ -n "$PREFILL_IP" && -n "$DECODE_IP" ]] || {
    echo "[config] failed to resolve data-plane IPs with DATA_NET=$DATA_NET" >&2
    return 1 2>/dev/null || exit 1
}

export ROCM_OPT_BASE_IMAGE="${ROCM_OPT_BASE_IMAGE:-rocm-llm-bench:glm52-v518-402df1e-2c71811}"
export V518_IMAGE="${V518_IMAGE:-infera/engine-sglang:glm52-v518-402df1e-2c71811}"
export IMAGE="${IMAGE:-$V518_IMAGE}"
export MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
export SERVED_MODEL="${SERVED_MODEL:-glm5.2-mxfp4}"
export MODEL_PREFIX="${MODEL_PREFIX:-glm5.2}"

export ETCD_PORT="${ETCD_PORT:-2379}"
export ROUTER_PORT="${ROUTER_PORT:-8000}"
# Keep fixed HTTP listeners outside Kubernetes' default NodePort range
# (30000-32767); an IPVS NodePort claim can intercept node-IP traffic even
# while a local bind and loopback health check both succeed.
export PREFILL_PORT="${PREFILL_PORT:-29001}"
export DECODE_PORT="${DECODE_PORT:-29002}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
export INFERA_NODEPORT_RANGE="${INFERA_NODEPORT_RANGE:-30000-32767}"

export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
export PREFILL_EP_SIZE="${PREFILL_EP_SIZE:-1}"
export DECODE_EP_SIZE="${DECODE_EP_SIZE:-1}"
export PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
export DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"
# Empty means "do not pass --context-length", so SGLang uses the model's own
# max_position_embeddings (1,048,576). This matches the rocm-llm-bench AgentX
# recipe, which caps neither the server nor the replay client.
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
export PREFILL_MEM_FRACTION="${PREFILL_MEM_FRACTION:-0.85}"
export DECODE_MEM_FRACTION="${DECODE_MEM_FRACTION:-0.85}"

# First v0.5.18 scan: one TP8/EP1 worker per leg, with no DP attention.
export PREFILL_DPA="${PREFILL_DPA:-0}"
export DECODE_DPA="${DECODE_DPA:-0}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export SPEC_STEPS="${SPEC_STEPS:-5}"
export SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-6}"
export SPEC_TOPK="${SPEC_TOPK:-1}"
# 5 steps / 6 draft tokens / 3.61 acceptance length are calibrated as a set.
# Deliberately omit ":" so SIMULATE_ACC_LEN= still disables simulation for correctness runs.
export SIMULATE_ACC_LEN="${SIMULATE_ACC_LEN-3.61}"
# Keep the production router contract even with one target per role. DP1 does
# not change placement, but disabling this also removes the KV event/snapshot
# plane and no longer validates the requested production configuration.
export ENABLE_KV_AWARE="${ENABLE_KV_AWARE:-1}"
export KV_PREFILL_OVERLAP_WEIGHT="${KV_PREFILL_OVERLAP_WEIGHT:-20.0}"
export KV_DECODE_OVERLAP_WEIGHT="${KV_DECODE_OVERLAP_WEIGHT:-2.0}"
# Infera's worker-side KV index publisher and snapshot endpoint. SGLang's
# scheduler publisher is allocated internally outside the ephemeral range.
export KV_PUB_PORT="${KV_PUB_PORT:-5557}"
export KV_SNAPSHOT_PORT="${KV_SNAPSHOT_PORT:-8801}"
export ENABLE_HICACHE="${ENABLE_HICACHE:-1}"
export PREFILL_ENABLE_HICACHE="${PREFILL_ENABLE_HICACHE:-$ENABLE_HICACHE}"
# SGLang PD decode forces ChunkCache. HiCache requires RadixCache, while
# decode-side RadixCache is rejected together with EAGLE/MTP in v0.5.18.
export DECODE_ENABLE_HICACHE="${DECODE_ENABLE_HICACHE:-0}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
export ENABLE_KVD="${ENABLE_KVD:-0}"
export KV_P2P_TRANSFER="${KV_P2P_TRANSFER:-mooncake}"

# The patched driver permits dma-buf registration on all eight index-affine
# ionic rails; destination affinity keeps ionic_N paired with remote ionic_N.
export RDMA_DEVICE="${RDMA_DEVICE:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
export MC_GID_INDEX="${MC_GID_INDEX:-1}"
export MOONCAKE_DISABLE_HIP_DMABUF="${MOONCAKE_DISABLE_HIP_DMABUF:-0}"
export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"
export MC_TE_FILTERS="${MC_TE_FILTERS:-$RDMA_DEVICE}"
export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
# The image entrypoint replaces its ABI-mismatched ionic provider with this
# host build. Both paths must match infera_inject_host_ionic.sh's contract.
export HOST_RDMA_LIB="${HOST_RDMA_LIB:-/lib/x86_64-linux-gnu/libionic.so}"
export HOST_RDMA_MOUNT="${HOST_RDMA_MOUNT:-/host-libionic/libionic.so}"
export NOFILE_ULIMIT="${NOFILE_ULIMIT:-65536:65536}"
export READY_TIMEOUT="${READY_TIMEOUT:-3600}"
export LEG_HEALTH_TRIES="${LEG_HEALTH_TRIES:-241}"
export GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-3600}"
export CONTAINER_STOP_TIMEOUT="${CONTAINER_STOP_TIMEOUT:-120}"
# This benchmark records the producer-push direction used by SGLang. It remains
# diagnostic evidence: changing it must not gate or invalidate correctness/AgentX.
export INFERA_PREFLIGHT_MOONCAKE_OPCODE="${INFERA_PREFLIGHT_MOONCAKE_OPCODE:-write}"

export AGENTX_CONCURRENCIES="${AGENTX_CONCURRENCIES:-8}"
export AGENTX_DURATION="${AGENTX_DURATION:-3600}"
export AGENTX_FAILED_REQUEST_THRESHOLD="${AGENTX_FAILED_REQUEST_THRESHOLD:-0.10}"
# Result metadata that /server_info cannot provide. agentx_point.sh derives and
# verifies observable topology, model, MTP, HiCache and GPU values at runtime.
export AGENTX_FRAMEWORK="${AGENTX_FRAMEWORK:-sglang}"
export AGENTX_PRECISION="${AGENTX_PRECISION:-fp4}"
export AGENTX_RUNNER_TYPE="${AGENTX_RUNNER_TYPE:-mi355x}"
export AGENTX_TOTAL_CPU_DRAM_GB="${AGENTX_TOTAL_CPU_DRAM_GB:-3023}"

# Correctness-case inputs.
export LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-250000}"
export GSM8K_CONCURRENT_REQUESTS="${GSM8K_CONCURRENT_REQUESTS:-$MAX_RUNNING_REQUESTS}"
export CORRECTNESS_EVAL_LIMIT="${CORRECTNESS_EVAL_LIMIT:-}"
export ROCM_LLM_BENCH_DIR="${ROCM_LLM_BENCH_DIR:-$BENCH_DIR/../../../rocm-llm-bench}"
export INFERENCEX_DIR="${INFERENCEX_DIR:-$ROCM_LLM_BENCH_DIR/third_party/InferenceX}"
export REMOTE_WORK_ROOT="${REMOTE_WORK_ROOT:-/mnt/m2m_nobackup/${USER:-liyingli}/infera_glm52}"

export RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/results}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-$RESULTS_DIR/$RUN_ID}"
