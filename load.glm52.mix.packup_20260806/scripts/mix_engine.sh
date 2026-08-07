#!/usr/bin/env bash
# what: launch ONE GLM-5.2 MIX worker (prefill + decode on the same 8 GPUs) via the
#       infera SGLang wrapper.
# why : this is the real launcher and it carries the tuned recipe. Every value that a
#       different site would change comes in as an env var.
# how : runs ON the node, docker-execs into $CTR. Called by mix_up.sh.
#
# Derived from examples/sglang_1p1d_glm5.2/engine/leg.sh. What was REMOVED, and why:
#   * --disaggregation-mode / -transfer-backend / -ib-device / -bootstrap-port
#     — there is no second leg; KV never crosses a wire.
#   * every MC_* / mooncake / dma-buf / GID env — same reason.
#   * the 0.70 prefill / 0.85 decode mem-fraction split — mix has ONE pool serving both
#     phases. GMU below is a single value and is the knob most likely to need tuning.
# What was KEPT verbatim, and why:
#   * the DSA-on-ROCm env block — mandatory on gfx950 or the model returns garbage
#   * --ep-size emitted UNCONDITIONALLY, outside the DPA branch (different axes)
#   * --disable-custom-all-reduce, independently of MTP (aiter AR deadlocks under
#     EAGLE verify; letting it follow MTP makes any MTP A/B two-variable)
#   * --kv-cache-dtype fp8_e4m3, --reasoning-parser glm45, --enable-cache-report
#
# Switches:  DPA=0|1  MTP=0|1  KVAWARE=0|1  KVD=0|1
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/mix_common.sh"

require_env MY_IP   "this node's data-plane IP"
require_env ETCD_IP "the node running etcd (this node, for mix)"
require_env MODEL   "model directory, inside \$MODEL_MOUNT"

TP="${TP:-8}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-256}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-128}"
HICACHE_GB="${HICACHE_GB:-32}"
KV_PUB_PORT="${KV_PUB_PORT:-5557}"
KV_SNAP_PORT="${KV_SNAP_PORT:-8801}"
PORT="${PORT:-$ENGINE_PORT}"
DPA="${DPA:-1}"; MTP="${MTP:-1}"; KVAWARE="${KVAWARE:-1}"; KVD="${KVD:-1}"
TAG="${TAG:-base}"
LOG="${LOG:-/tmp/glm52_mix_${TAG}.log}"

# ONE pool for both phases. In PD these were 0.70 (prefill, activation headroom) and 0.85
# (decode, KV pool). Mix cannot have both; 0.80 is the midpoint we start from and A/B.
# Direction if it fails: prefill activation OOM (HSA_STATUS_ERROR_OUT_OF_RESOURCES at LOW
# token usage) -> LOWER. Decode retract / get_cpu_copy NotImplementedError -> RAISE.
GMU="${GMU:-0.80}"

# A GLOBAL budget that SGLang divides by dp_size ONLY when DP-attention is on. One value
# serves both modes; do NOT hardcode a per-rank number in a DPA-off branch.
CHUNK="${CHUNK:-65536}"

export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1

NIC="${NIC:-$(docker exec "$CTR" bash -c "ip -o -4 addr show | awk '\$4 ~ /^$MY_IP\// {print \$2; exit}'" 2>/dev/null | tr -d '\r')}"
[ -n "$NIC" ] || die "could not derive the NIC holding $MY_IP inside $CTR — set NIC explicitly"

# ---- GLM-5.2 DSA-on-ROCm recipe (MANDATORY on gfx950) --------------------------------
# Without these the model still serves and still returns 200s — it just returns garbage,
# because the sparse-attention indexer takes a path not ported to ROCm.
DSA_ENV="SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0 \
SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1"
# Stable block hashes -> stable kvd keys across restarts.
DSA_ENV="$DSA_ENV PYTHONHASHSEED=0"
[ "$DPA" = "1" ] && DSA_ENV="$DSA_ENV SGLANG_DP_USE_GATHERV=1"

# ---- args ----------------------------------------------------------------------------
# --ep-size is OUTSIDE the DPA branch on purpose: expert parallelism and attention
# parallelism are different axes. Moving it inside silently collapses the MoE whenever
# DPA is off, and then no latency delta is attributable to either.
DP_ARGS=(--ep-size "$TP")
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)
  [ "${DELAYER:-1}" = "1" ] \
    && DP_ARGS+=(--enable-prefill-delayer --prefill-delayer-max-delay-ms "${PREFILL_DELAY_MS:-5000}")
fi

INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT" --kv-snapshot-port "$KV_SNAP_PORT")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi

# kvd is LEGAL on mix: kvd_wiring._skip_kvd_on_decode_leg skips only when
# disaggregation_mode == "decode", and SGLang's storage prefetch runs on the aggregated
# branch of Scheduler._add_request_to_queue. --hicache-size is ABSOLUTE GB, deliberately.
[ "$KVD" = "1" ] && INFERA_ARGS+=(--infera-kvd-socket "$KVD_SOCK" --hicache-size "$HICACHE_GB")

MTP_ARGS=()
if [ "$MTP" = "1" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE
            --speculative-num-steps "${SPEC_STEPS:-3}"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "${SPEC_DRAFT:-4}")
fi

CAR_ARGS=()
[ "${CUSTOM_AR:-0}" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

EXTRA_ARGS=(--enable-cache-report)

log "mix worker on $MY_IP:$PORT — tp=$TP dpa=$DPA mtp=$MTP kvaware=$KVAWARE kvd=$KVD gmu=$GMU chunk=$CHUNK ctx=$CTX tag=$TAG"

docker exec -d "$CTR" env \
  HIP_VISIBLE_DEVICES="$GPUS" \
  NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1 \
  SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP" \
  SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC" \
  INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}" \
  bash -c "export $DSA_ENV; python3 -m infera.engine.sglang \
    --model-path '$MODEL' --served-model-name '$SERVED' --tp-size $TP --trust-remote-code \
    --host '$MY_IP' --port $PORT \
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static $GMU --context-length $CTX \
    --chunked-prefill-size $CHUNK --cuda-graph-max-bs $CUDA_GRAPH_BS \
    --max-running-requests $MAX_RUNNING --watchdog-timeout ${WATCHDOG:-3600} \
    --reasoning-parser glm45 \
    ${INFERA_ARGS[*]} ${DP_ARGS[*]} ${MTP_ARGS[*]} ${CAR_ARGS[*]} ${EXTRA_ARGS[*]} \
    > '$LOG' 2>&1"

log "mix worker launching -> $LOG in $CTR (cold start is minutes: weights + graph capture)"
