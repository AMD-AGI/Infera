#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: bring up the whole 1P1D deployment — containers, etcd, UMBP, both PD legs, router.
# why : one command from the wrapper. The router pairs the two legs out of etcd, so there is
#       no static worker list to keep in sync.
# how : DO NOT run this directly — run it through cluster/<your-cluster>.sh, which supplies every
#       site-specific value. Reaches each node via `on` (local, or $SSH_CMD for a remote node).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env DECODE_NODE
require_env PREFILL_IP;   require_env DECODE_IP
require_env KIT_DIR "path to this kit, identical on both nodes"
require_env INFERA_IMAGE; require_env MODEL; require_env MODEL_MOUNT
require_env PREFILL_GPUS; require_env DECODE_GPUS
require_env MR_MODE; require_env RDMA_IB_DEVICES; require_env MC_GID_INDEX
ENGINE_READY_TIMEOUT="${READY_TIMEOUT:-3600}"

# The env every remote invocation needs. Kept in one string so the two legs cannot drift.
COMMON_ENV="CTR=$CTR KIT_DIR=$KIT_DIR INFERA_IMAGE=$INFERA_IMAGE MODEL=$MODEL MODEL_MOUNT=$MODEL_MOUNT \
SERVED=$SERVED TOKENIZER=${TOKENIZER:-$MODEL} ETCD_IP=$PREFILL_IP ETCD_PORT=$ETCD_PORT \
ROUTER_PORT=$ROUTER_PORT PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT \
LOG_DIR=$LOG_DIR UMBP_DIR=$UMBP_DIR MR_MODE=$MR_MODE RDMA_IB_DEVICES=$RDMA_IB_DEVICES \
MC_GID_INDEX=$MC_GID_INDEX ${RDMA_GPU_NICS:+RDMA_GPU_NICS=$RDMA_GPU_NICS} \
READY_TIMEOUT=$ENGINE_READY_TIMEOUT ${GMU:+GMU=$GMU} ${D_PSI:+D_PSI=$D_PSI} \
${GOLDEN_AL:+GOLDEN_AL=$GOLDEN_AL} ${EXTRA_ENGINE_ARGS:+EXTRA_ENGINE_ARGS=\"$EXTRA_ENGINE_ARGS\"}"

log "=== 1/5 containers ==="
nodes=("$PREFILL_NODE"); [ "$DECODE_NODE" != "$PREFILL_NODE" ] && nodes+=("$DECODE_NODE")
for h in "${nodes[@]}"; do
  on "$h" "$COMMON_ENV bash -c 'source $KIT_DIR/common.sh; start_container'"
done

log "=== 2/5 etcd (prefill node) ==="
on "$PREFILL_NODE" "$COMMON_ENV bash -c 'source $KIT_DIR/common.sh; start_etcd $PREFILL_IP'"

log "=== 3/5 UMBP host-DRAM KV store (prefill node) ==="
if [ "${UMBP:-1}" = "1" ] && [ "${UMBP_MODE:-standalone}" = "embedded" ]; then
  log "UMBP embedded: each prefill DP rank keeps its own ${UMBP_GB:-449} GB / ranks DRAM pool in-process"
elif [ "${UMBP:-1}" = "1" ]; then
  # A GEM prefill KV cache is mapped by the server through the GEM allocator's IPC hooks.
  preload=""; [ "${PREFILL_MR_MODE:-$MR_MODE}" = gem ] && preload="UMBP_PRELOAD=/usr/local/lib/libhip_gem_alloc.so"
  on "$PREFILL_NODE" "$COMMON_ENV UMBP_GB=${UMBP_GB:-449} $preload bash -c 'source $KIT_DIR/common.sh; start_umbp'"
else
  log "UMBP=0: prefill keeps only its GPU prefix cache"
fi

log "=== 4/5 legs ==="
# Launch both legs before waiting on either: they load the weights concurrently, and
# serialising the waits doubles the bring-up for no reason.
# PREFILL_/DECODE_ MR_MODE and GMU override the shared values per leg (later assignments win).
on "$PREFILL_NODE" "$COMMON_ENV ROLE=prefill MY_IP=$PREFILL_IP GPUS=$PREFILL_GPUS UMBP=${UMBP:-1} \
  UMBP_MODE=${UMBP_MODE:-standalone} UMBP_GB=${UMBP_GB:-449} \
  ${PREFILL_MR_MODE:+MR_MODE=$PREFILL_MR_MODE} ${PREFILL_GMU:+GMU=$PREFILL_GMU} bash $KIT_DIR/engine/leg.sh"
on "$DECODE_NODE" "$COMMON_ENV ROLE=decode MY_IP=$DECODE_IP GPUS=$DECODE_GPUS SPEC_MODE=${SPEC_MODE:-block} \
  ${DECODE_MR_MODE:+MR_MODE=$DECODE_MR_MODE} ${DECODE_GMU:+GMU=$DECODE_GMU} bash $KIT_DIR/engine/leg.sh"

# Poll /health from INSIDE each node's container; never grep a log for a ready line.
log "waiting for both legs (cold start is minutes: weights, Engram tables, CUDA-graph capture)"
wait_leg(){
  local h="$1" ip="$2" port="$3" role="$4" tries=$(( (ENGINE_READY_TIMEOUT + 14) / 15 ))
  for i in $(seq 1 "$tries"); do
    if on "$h" "docker exec $CTR curl -sf -m5 http://$ip:$port/health >/dev/null 2>&1"; then
      log "  $role ($ip:$port) serving after $((i * 15))s"; return 0
    fi
    if ! on "$h" "docker exec $CTR pgrep -f -- 'infera.engine.vllm.*--port $port( |\$)' >/dev/null"; then
      grep -aE 'Error|error:|Traceback' "$LOG_DIR/$role.log" | tail -5; warn "  $role leg died"; return 1
    fi
    sleep 15
  done
  warn "  $role ($ip:$port) did NOT come up in $((tries * 15))s"; return 1
}
wait_leg "$PREFILL_NODE" "$PREFILL_IP" "$PREFILL_PORT" prefill & wp=$!
wait_leg "$DECODE_NODE" "$DECODE_IP" "$DECODE_PORT" decode & wd=$!
wait $wp || die "prefill leg never became ready ($LOG_DIR/prefill.log)"
wait $wd || die "decode leg never became ready ($LOG_DIR/decode.log)"

log "=== 5/5 router (prefill node) ==="
# Last: it registers the workers it finds in etcd, so its first health check already reflects a
# paired deployment.
on "$PREFILL_NODE" "$COMMON_ENV ${KV_PREFILL_W:+KV_PREFILL_W=$KV_PREFILL_W} ${KV_DECODE_W:+KV_DECODE_W=$KV_DECODE_W} \
  ${SESSION_AFFINITY:+SESSION_AFFINITY=$SESSION_AFFINITY} bash -c 'source $KIT_DIR/common.sh; start_router $PREFILL_IP'"

log "up. endpoint: http://$PREFILL_IP:$ROUTER_PORT   logs: $LOG_DIR"
log "verify with:  bash cluster/<your-cluster>.sh smoke"
