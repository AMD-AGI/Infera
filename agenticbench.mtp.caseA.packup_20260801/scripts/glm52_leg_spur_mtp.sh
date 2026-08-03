#!/bin/bash
# GLM-5.2-MXFP4 cross-node PD leg on SPUR with the FULL feature set:
# kvaware + kvd + MTP(EAGLE) + PD(mooncake) + DP-attention.
# Runs INSIDE the engine container. Launched through the INFERA WRAPPER so the
# kvaware/kvd wiring is actually in the path.
#
# THIS IS A FUSION of the two sanctioned kits. Neither runs here as-is:
#
#   * glm52.merged_branch_image.packup_20260801/scripts/glm52_leg.sh has the MTP +
#     kvd + kvaware wiring validated on the merged branch -- but it AUTO-DISCOVERS
#     ionic NICs and `exit 1`s when there are none, hardcodes MC_GID_INDEX=1 and
#     DMABUF=0. That is the VULTR fabric; on spur it dies at the NIC scan.
#   * agenticbench.glm52.spur.packup_20260731/scripts/glm52_leg_spur.sh has the
#     correct spur transport and the Case A sizing -- but it deliberately carries
#     NO --speculative-* flags at all (MTP was off by operator instruction).
#
# So: spur transport + Case A sizing from the latter, MTP block + CUSTOM_AR
# handling from the former. Getting the transport wrong loses RDMA silently
# (MC_FORCE_TCP -- correct but slow, and it looks fine); getting the MTP block
# wrong measures a deployment without the feature under test.
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
# Case A inputs are p50 74K / p99 235K. At ctx=131072 the clamp truncates 16.1%
# of the distribution (measured: probe p90 AND p99 both pinned at exactly
# 131,072); at 262144 only ~1.4% clamps.
CTX="${CTX:-262144}"
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
# --- MTP (EAGLE spec-dec). Decode leg only unless PREFILL_MTP=1. ------------
# Every deadlock the converged kit measured was under decode-only MTP, and that
# is the configuration the branch's G1 gate validates.
MTP="${MTP:-0}"
# ---- custom all-reduce: INDEPENDENT of MTP ----
# The aiter custom all-reduce kernel deadlocks on gfx942/gfx950 during EAGLE
# verify at high concurrency (sglang #28815 / #31071 / PR #31478). That is a
# defect in the aiter kernel itself on this arch -- it is NOT one of the three
# DSA/DP-attention/draft-graph rank-divergence bugs the branch patches fix, and
# fixing those does not make this kernel safe. The converged kit
# (glm52.mxfp4.spur.mooncake.packup_20260731_main_converged) states it three
# times over, and records that the custom all-reduce path "remains unexercised".
#
# It defaults OFF (i.e. --disable-custom-all-reduce is passed) for BOTH arms.
# This used to follow MTP here, which meant MTP=0 silently RE-ENABLED a
# known-broken kernel and turned "MTP on vs off" into a TWO-variable comparison
# -- exactly the trap the converged kit's leg script calls out and removed. The
# A/B below differs in MTP and nothing else.
CUSTOM_AR="${CUSTOM_AR:-0}"
# ---- mem-fraction-static: prefill 0.80, NOT 0.88 --------------------------
# At 0.88 the prefill leg aborts with
#     rocdevice.cpp:3582 HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Fatal Python error: Aborted
# while `token usage` reads 0.01-0.05 -- the KV pool is nearly EMPTY, so this is
# NOT KV exhaustion. It is DP-attention runtime ACTIVATION memory: at dp8 every
# rank holds its own 8192-token chunk activations, and a 155K prompt is 19
# chunks; the transient peak exceeds what 1 - mem_fraction_static leaves outside
# the static reservation.
#
# The direction is counter-intuitive and worth stating: prefill activation OOM
# is fixed by LOWERING mem-fraction-static -- the OPPOSITE of the decode-side
# retract fix (raise it, for more KV room). Diagnose by phase:
#     decode   retract / get_cpu_copy NotImplementedError  -> RAISE
#     prefill  HSA_STATUS_ERROR_OUT_OF_RESOURCES / Aborted -> LOWER
#
# 0.80 is first-hand verified on the vultr sibling run of this same branch and
# image (fixlen.glm52.fullfeature.packup_20260801/patches/0002): 0.88 crashed at
# ISL 155K x conc 32; at 0.80 that point ran 32/32 clean and c64/c128 followed
# with zero HSA_STATUS_ERROR. Cost: KV pool -13 %, activation headroom +68 %.
# Decode stays 0.85 -- it never crashed, and moving both would make this a
# two-variable change.
if [ "$ROLE" = "prefill" ]; then GMU="${GMU:-0.80}"; else GMU="${GMU:-0.85}"; fi
LOG="${LOG:-/shared_nfs/yihou_agbench_mtp/${ROLE}.log}"
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

# ---- MTP: EAGLE steps=3 (decode leg by default; PREFILL_MTP=1 adds prefill) --
MTP_ARGS=()
if [ "$MTP" = "1" ] && { [ "$ROLE" = "decode" ] || [ "${PREFILL_MTP:-0}" = "1" ]; }; then
  MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps "${SPEC_STEPS:-3}" \
            --speculative-eagle-topk 1 --speculative-num-draft-tokens "${SPEC_DRAFT:-4}" \
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi

CAR_ARGS=()
[ "$CUSTOM_AR" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

# --enable-cache-report: without it the server does not populate
# usage.prompt_tokens_details.cached_tokens, agent-bench's cache_hit% reads 0,
# and the deliverable loses its cache column entirely.
EXTRA=(--enable-cache-report)

echo "[agbench-spur-mtp] role=$ROLE ip=$MY_IP:$PORT nic=$NIC ibdev=$IBDEV gid=$GID dpa=$DPA mtp=$MTP car=$CUSTOM_AR kvaware=$KVAWARE kvd=$KVD dmabuf=$DMABUF gmu=$GMU chunk=$CHUNK ctx=$CTX -> $LOG"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$GMU" --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --max-running-requests "$MAX_RUNNING" --watchdog-timeout 3600 \
  "${INFERA_ARGS[@]}" "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${CAR_ARGS[@]}" \
  "${ROLE_ARGS[@]}" "${EXTRA[@]}" ${EXTRA_ARGS:-} > "$LOG" 2>&1
