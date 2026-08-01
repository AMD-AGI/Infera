#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg on SPUR, launched through the INFERA WRAPPER so
# kv-aware routing and kvd are actually in the path. Runs INSIDE the agbench container.
#
# THIS IS A FUSION. Neither sanctioned kit's leg script runs here as-is:
#
#   * pr.final/scripts/glm52_leg.sh has the kvaware/kvd wiring, but auto-discovers
#     ionic NICs (and `exit 1`s if none), hardcodes MC_GID_INDEX=1 and DMABUF=0.
#     That is the VULTR fabric. On spur it would either die or silently pick the
#     wrong NIC.
#   * final_deliverable/scripts/pd_leg_exp.sh has the correct spur transport, but
#     launches `python3 -m sglang.launch_server`, which bypasses the infera wrapper
#     entirely -- no kv-events, no kvd, i.e. none of the features under test.
#
# So: infera wrapper entry + INFERA_ARGS from the former; mlx5/GID-3/dma-buf
# transport block from the latter. Getting this wrong loses either RDMA (silent
# TCP fallback -- correct but slow, and it looks fine) or the features being
# benchmarked.
#
# MTP IS DELIBERATELY ABSENT. The DPA+PD+MTP fix is not merged with kv-aware yet
# (operator instruction, 2026-07-31). No --speculative-* flags anywhere below.
set -u
ROLE="${ROLE:?ROLE=prefill|decode}"
MY_IP="${MY_IP:?MY_IP=this node ens3 IP}"
P_IP="${P_IP:?P_IP=prefill ens3 IP (bootstrap host)}"
ETCD_IP="${ETCD_IP:?}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ETCD_PORT="${ETCD_PORT:-2379}"
TP="${TP:-8}"
DPA="${DPA:-1}"
# The bench's context length, NOT the kits' 32768. Case A inputs are p50 74K /
# p99 235K; the engine must cover the clamp plus output.
CTX="${CTX:-131072}"
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
DELAYER="${DELAYER:-1}"
PREFILL_DELAY_MS="${PREFILL_DELAY_MS:-5000}"
NIC="${NIC:-ens3}"                    # the single mlx5 netdev with an IP on spur
IBDEV="${IBDEV:-mlx5_0}"              # spur KV NIC = mlx5 (has ODP)
GID="${GID:-3}"                       # mlx5 RoCEv2 routable GID index on spur
DMABUF="${DMABUF:-1}"                 # spur: dmabuf ON (no peermem)
# --- features under test ---
KVAWARE="${KVAWARE:-1}"
KVD="${KVD:-1}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
HICACHE_GB="${HICACHE_GB:-32}"        # ABSOLUTE. Never --hicache-ratio (355 GB/rank once).
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.88}"; else GMU="${GMU:-0.85}"; fi
LOG="${LOG:-/shared_nfs/yihou_agentbench/${ROLE}.log}"
mkdir -p "$(dirname "$LOG")"

ISL="${ISL:-8192}"
if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"; else CHUNK="${CHUNK:-8192}"; fi

# ---- 408 GB checkpoint, two legs off one filesystem: the 1800 s default is tight.
export INFERA_SGLANG_READY_TIMEOUT="${INFERA_SGLANG_READY_TIMEOUT:-3600}"

# ---- spur transport: no peermem, so dma-buf via mlx5(ODP) is the ONLY GPUDirect path.
if [ "$DMABUF" = "1" ]; then export MOONCAKE_DISABLE_HIP_DMABUF=0; else export MOONCAKE_DISABLE_HIP_DMABUF=1; fi

# ---- mooncake RDMA env, FORCED onto the single mlx5 NIC (not the 8 ionic rails) ----
export MC_GID_INDEX="$GID" MC_DISABLE_HIP_TRANSPORT=1
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS="$IBDEV"
unset MC_ENABLE_HIP_TRANSPORT
export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

# ---- GLM-5.2 DSA-ROCm recipe (mandatory on gfx950) ----
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1

[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1
# Stable block hashes -> stable kvd keys across restarts (and across the two legs).
export PYTHONHASHSEED=0

ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake \
           --disaggregation-ib-device "$IBDEV")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")
  if [ "$ROLE" = "prefill" ] && [ "$DELAYER" = "1" ]; then
    DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "$PREFILL_DELAY_MS")
  fi
fi

# infera wrapper args: etcd discovery so the infera router pairs the legs.
INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)

if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT"
                --kv-snapshot-port "$KV_SNAP_PORT")
else
  # NOTE: this also kills kvd on the DECODE leg. A PD decode leg sets
  # disable_radix_cache=True itself and sglang rejects hierarchical cache
  # alongside it; what legalises kvd there is
  # --disaggregation-decode-enable-radix-cache, which infera appends ONLY when
  # kv-events are on. The two switches are not independent.
  INFERA_ARGS+=(--no-enable-kv-events)
fi

# kvd: the wrapper probes the daemon, then appends --enable-hierarchical-cache +
# --hicache-storage-backend dynamic + the InferaKvdBackend extra-config.
[ "$KVD" = "1" ] && INFERA_ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

# --enable-cache-report: without it the server does not populate
# usage.prompt_tokens_details.cached_tokens, agent-bench's cache_hit% reads 0,
# and the deliverable loses its cache column entirely.
# --disable-custom-all-reduce: the aiter custom all-reduce kernel deadlocks on
# gfx942/gfx950 at high concurrency. Kept even without MTP.
EXTRA=(--enable-cache-report --disable-custom-all-reduce)

echo "[agbench-spur] role=$ROLE ip=$MY_IP:$PORT nic=$NIC ibdev=$IBDEV gid=$GID dpa=$DPA mtp=OFF kvaware=$KVAWARE kvd=$KVD dmabuf=$DMABUF gmu=$GMU chunk=$CHUNK ctx=$CTX -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${INFERA_ARGS[@]}" "${DP_ARGS[@]}" "${ROLE_ARGS[@]}" "${EXTRA[@]}" ${EXTRA_ARGS:-} > "$LOG" 2>&1
