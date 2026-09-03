#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE GLM-5.2 PD leg (prefill|decode) through the infera SGLang wrapper.
# why : this is the real launcher. It carries the tuned recipe and NOTHING site-specific —
#       every address, path, NIC name and GID comes in as an env var from
#       cluster/<your-cluster>.sh. If you need to adapt to your cluster, edit THAT file,
#       not this one.
# how : runs ON the node, docker-execs into $CTR. Called by engine/up.sh.
#
# Feature switches (all default to the recommended production values):
#   DPA=0|1     DP-attention on this leg
#   MTP=0|1     EAGLE speculative decoding (decode leg only unless PREFILL_MTP=1)
#   KVAWARE=0|1 publish KV events so the router can route by cache locality
#   KVD=0|1     wire the infera-kvd HiCacheStorage backend (L2 host RAM + L3)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

ROLE="${ROLE:?ROLE=prefill|decode}"
require_env MY_IP   "this node's data-plane IP (the one peers and the router can reach)"
require_env ETCD_IP "the node running etcd (by convention the prefill node)"
require_env MODEL   "model directory, inside \$MODEL_MOUNT"
require_env RDMA_IB_DEVICES "comma-separated RDMA device(s) mooncake may use, e.g. mlx5_0"

TP="${TP:-8}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
HICACHE_GB="${HICACHE_GB:-32}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"

if [ "$ROLE" = "prefill" ]; then
  PORT="${PORT:-$PREFILL_PORT}"; DPA="${DPA:-0}"; MTP="${MTP:-0}"; KVD="${KVD:-1}"
else
  PORT="${PORT:-$DECODE_PORT}";  DPA="${DPA:-1}"; MTP="${MTP:-1}"; KVD="${KVD:-0}"
fi
KVAWARE="${KVAWARE:-1}"
LOG="${LOG:-/tmp/glm52_${ROLE}.log}"

# Role-asymmetric, and the direction is counter-intuitive: a prefill OOM is fixed by LOWERING
# this, a decode retract by raising it. Also coupled to DPA and to the router policy.
# 0.70/0.85 is safe across every combination measured on MI355X. Diagnosis: README note 3.
GMU_PREFILL="${GMU_PREFILL:-0.70}"
GMU_DECODE="${GMU_DECODE:-0.85}"
[ "$ROLE" = "prefill" ] && GMU="${GMU:-$GMU_PREFILL}" || GMU="${GMU:-$GMU_DECODE}"

# A GLOBAL budget that SGLang divides by dp_size only when DP-attention is on. One value
# serves both modes; DO NOT hardcode a per-rank number in a DPA-off branch — that cuts the
# global budget 8x, silently. Why the trap is expensive: README note 2.
CHUNK="${CHUNK:-65536}"

# ---- transport env ------------------------------------------------------------------------
# All from cluster/<wrapper>.sh, filled in from the preflight report. Do not guess: a wrong
# GID index fails loudly, but a wrong dma-buf/peer-mem choice fails SILENTLY — doubled KV
# pool, or a 5-20x slower TCP fallback. See cluster/README.md section 3.
export MOONCAKE_DISABLE_HIP_DMABUF="${MOONCAKE_DISABLE_HIP_DMABUF:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:?MC_GID_INDEX is required — read it off the preflight report}"
export MC_DISABLE_HIP_TRANSPORT=1
unset MC_ENABLE_HIP_TRANSPORT
# MC_TE_FILTERS pins mooncake to named device(s). Required in the dma-buf mode so a non-ODP
# rail is never picked (it would pin and double the KV pool); harmless to leave unset when a
# peer-mem module is loaded and every rail can carry KV.
[ -n "${MC_TE_FILTERS:-}" ] && export MC_TE_FILTERS
[ "${RDMAV_FORK_SAFE:-0}" = "1" ] && export RDMAV_FORK_SAFE=1
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1

# Bind the engine and its collectives to the data-plane NIC. NIC is derived from MY_IP rather
# than named, so the same script works on a node whose interface has a different name.
NIC="${NIC:-$(docker exec "$CTR" bash -c "ip -o -4 addr show | awk '\$4 ~ /^$MY_IP\// {print \$2; exit}'" 2>/dev/null | tr -d '\r')}"
[ -n "$NIC" ] || die "could not derive the NIC holding $MY_IP inside $CTR — set NIC explicitly"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
# A 400 GB+ checkpoint over shared storage, two legs reading it at once: the defaults are tight.
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${BOOTSTRAP_TIMEOUT:-1800}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${BOOTSTRAP_TIMEOUT:-1800}"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"

# ---- GLM-5.2 on ROCm -------------------------------------------------------------------
# The correctness-critical one, SGLANG_OPT_USE_TOPK_V2=0, is not here: infera.engine.sglang
# applies it itself (infera/engine/rocm_dsa_env.py). See README note 5.
export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1
# Stable block hashes -> stable kvd keys across restarts, so an L3 entry written by one run is
# readable by the next.
export PYTHONHASHSEED=0

# ---- args --------------------------------------------------------------------------------
ROLE_ARGS=(--disaggregation-mode "$ROLE"
           --disaggregation-transfer-backend mooncake
           --disaggregation-ib-device "$RDMA_IB_DEVICES")
[ "$ROLE" = "prefill" ] && ROLE_ARGS+=(--disaggregation-bootstrap-port "$BOOTSTRAP_PORT")

# --ep-size is passed UNCONDITIONALLY, outside the DPA branch: expert parallelism and
# attention parallelism are different axes. DO NOT move it inside the `if` — that silently
# collapses the MoE too whenever DPA is off. See README note 1.
DP_ARGS=(--ep-size "$TP")
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)
  # The prefill delayer batches arrivals so a DP step is not wasted on one short request.
  # It only makes sense where there are DP ranks to fill.
  [ "$ROLE" = "prefill" ] && [ "${DELAYER:-1}" = "1" ] \
    && DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "${PREFILL_DELAY_MS:-5000}")
fi

# etcd discovery, so the router pairs the two legs with no static worker list.
INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)

if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT" --kv-snapshot-port "$KV_SNAP_PORT")
else
  # NOTE: this also disables kvd on the DECODE leg — the two switches are not independent.
  # What legalises kvd there is --disaggregation-decode-enable-radix-cache, which infera
  # appends only when KV events are on. See README note 6.
  INFERA_ARGS+=(--no-enable-kv-events)
fi

# kvd: the wrapper probes the daemon, then appends --enable-hierarchical-cache,
# --hicache-storage-backend and the backend's extra-config for you. --hicache-size is an
# ABSOLUTE size in GB, deliberately — the ratio-based default can wedge a node. README note 7.
[ "$KVD" = "1" ] && INFERA_ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

# MTP (EAGLE). Decode-only by default: that is the configuration these settings were
# validated in. Acceptance length should land around 2-3; a steady 4.00 means the draft model
# is predicting a repetition loop, not that speculation is working well.
MTP_ARGS=()
if [ "$MTP" = "1" ] && { [ "$ROLE" = "decode" ] || [ "${PREFILL_MTP:-0}" = "1" ]; }; then
  MTP_ARGS=(--speculative-algorithm EAGLE
            --speculative-num-steps "${SPEC_STEPS:-3}"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "${SPEC_DRAFT:-4}"
            --num-reserved-decode-tokens "${RESERVED_TOK:-256}")
fi

# Disabled by default and INDEPENDENTLY of MTP — the aiter kernel deadlocks on gfx942/gfx950
# during EAGLE verify. DO NOT let this switch follow MTP: that makes any "MTP on vs off"
# comparison two-variable. Set CUSTOM_AR=1 only if your image carries the fix. README note 4.
CAR_ARGS=()
[ "${CUSTOM_AR:-0}" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

# --enable-cache-report populates usage.prompt_tokens_details.cached_tokens. Without it every
# client-side cache-hit metric reads 0 and a prefix-reuse target cannot be checked at all.
EXTRA_ARGS=(--enable-cache-report)

log "$ROLE on $MY_IP:$PORT — tp=$TP dpa=$DPA mtp=$MTP kvaware=$KVAWARE kvd=$KVD gmu=$GMU chunk=$CHUNK ctx=$CTX nic=$NIC ib=$RDMA_IB_DEVICES"

# Pass the transport/recipe env explicitly rather than relying on `docker exec`'s environment:
# the variables above are exported in THIS shell, on the host, not inside the container.
docker exec -d "$CTR" env \
  HIP_VISIBLE_DEVICES="$GPUS" \
  MOONCAKE_DISABLE_HIP_DMABUF="$MOONCAKE_DISABLE_HIP_DMABUF" \
  MC_GID_INDEX="$MC_GID_INDEX" MC_DISABLE_HIP_TRANSPORT=1 \
  ${MC_TE_FILTERS:+MC_TE_FILTERS="$MC_TE_FILTERS"} \
  ${RDMAV_FORK_SAFE:+RDMAV_FORK_SAFE="$RDMAV_FORK_SAFE"} \
  NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1 \
  SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP" \
  SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC" \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="$SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT" \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT="$SGLANG_DISAGGREGATION_WAITING_TIMEOUT" \
  INFERA_SGLANG_READY_TIMEOUT="$INFERA_SGLANG_READY_TIMEOUT" \
  SGLANG_USE_AITER=1 \
  SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 PYTHONHASHSEED=0 \
  ${SGLANG_DP_USE_GATHERV:+SGLANG_DP_USE_GATHERV=1} \
  bash -c "python3 -m infera.engine.sglang \
    --model-path '$MODEL' --served-model-name '$SERVED' --tp-size $TP --trust-remote-code \
    --host '$MY_IP' --port $PORT \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static $GMU --context-length $CTX \
    --chunked-prefill-size $CHUNK --cuda-graph-max-bs $CUDA_GRAPH_BS \
    --max-running-requests $MAX_RUNNING --watchdog-timeout ${WATCHDOG:-3600} \
    --reasoning-parser glm45 \
    ${INFERA_ARGS[*]} ${DP_ARGS[*]} ${MTP_ARGS[*]} ${CAR_ARGS[*]} \
    ${ROLE_ARGS[*]} ${EXTRA_ARGS[*]} > '$LOG' 2>&1"

log "$ROLE launching -> $LOG in $CTR (cold start is minutes: weights + graph capture)"
