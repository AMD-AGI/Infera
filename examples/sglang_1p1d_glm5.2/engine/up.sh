#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: bring up the whole 1P1D deployment — containers, etcd, kvd, router, both PD legs.
# why : one command from the wrapper. The router pairs the two legs out of etcd, so there is
#       no static worker list to keep in sync.
# how : DO NOT run this directly — run it through cluster/<your-cluster>.sh, which supplies
#       every site-specific value. Reaches each node via $SSH_CMD.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env DECODE_NODE
require_env PREFILL_IP;   require_env DECODE_IP
require_env KIT_DIR "path to this kit, identical on both nodes"
require_env INFERA_IMAGE; require_env MODEL; require_env MODEL_MOUNT
require_env RDMA_IB_DEVICES; require_env MC_GID_INDEX

# On a scheduler that blocks ssh to compute nodes, set SSH_CMD to whatever runs a command on
# a node. Invoked as: $SSH_CMD <node> <command>.
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; $SSH_CMD "$h" "$*"; }

# The env every remote invocation needs. Kept in one string so the two legs cannot drift.
# PREFILL_MTP is forwarded as well as consumed below: leg.sh reads it directly to decide
# whether a prefill leg may emit MTP args, so passing only MTP= leaves that gate always shut.
COMMON_ENV="CTR=$CTR INFERA_IMAGE=$INFERA_IMAGE MODEL=$MODEL MODEL_MOUNT=$MODEL_MOUNT \
SERVED=$SERVED TOKENIZER=${TOKENIZER:-$MODEL} ETCD_IP=$PREFILL_IP ETCD_PORT=$ETCD_PORT \
ROUTER_PORT=$ROUTER_PORT PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT \
BOOTSTRAP_PORT=$BOOTSTRAP_PORT KVD_SOCK=$KVD_SOCK TP=${TP:-8} CTX=${CTX:-262144} \
CHUNK=${CHUNK:-65536} KVAWARE=${KVAWARE:-1} PREFILL_MTP=${PREFILL_MTP:-0} \
RDMA_IB_DEVICES=$RDMA_IB_DEVICES MC_GID_INDEX=$MC_GID_INDEX \
MOONCAKE_DISABLE_HIP_DMABUF=${MOONCAKE_DISABLE_HIP_DMABUF:-1} \
${MC_TE_FILTERS:+MC_TE_FILTERS=$MC_TE_FILTERS} \
${RDMAV_FORK_SAFE:+RDMAV_FORK_SAFE=$RDMAV_FORK_SAFE} \
${HOST_RDMA_LIB:+HOST_RDMA_LIB=$HOST_RDMA_LIB} \
${ENTRYPOINT_KEEP:+ENTRYPOINT_KEEP=$ENTRYPOINT_KEEP} \
${GMU_PREFILL:+GMU_PREFILL=$GMU_PREFILL} ${GMU_DECODE:+GMU_DECODE=$GMU_DECODE}"

log "=== 1/4 containers ==="
for h in "$PREFILL_NODE" "$DECODE_NODE"; do
  on "$h" "$COMMON_ENV bash -c 'source $KIT_DIR/common.sh; start_container'"
done

log "=== 2/4 etcd (prefill node) + kvd ==="
on "$PREFILL_NODE" "$COMMON_ENV bash -c 'source $KIT_DIR/common.sh; start_etcd $PREFILL_IP'"

# kvd runs only where a leg enables it. Stage a script FILE and nohup it — a detached
# `docker exec -d ... bash -lc '<cmd>'` login shell exits and takes the child with it, leaving
# no process, no log, no error. --max-bytes/--long-bytes are absolute caps: README note 7.
start_kvd(){
  local h="$1"
  on "$h" "docker exec $CTR bash -c \"mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && \
printf '%s\\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK \
--max-bytes ${KVD_MAX_BYTES:-64G} --long-path /tmp/kvd-long --long-bytes ${KVD_LONG_BYTES:-64G} \
--log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh\""
  on "$h" "docker exec -d $CTR bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'"
  sleep 15
  on "$h" "docker exec $CTR bash -c \"test -S $KVD_SOCK && echo '  kvd socket OK on $h' || { echo '  KVD FAILED on $h'; tail -20 /tmp/kvd.log; }\""
}
if [ "${PREFILL_KVD:-1}" = "1" ]; then start_kvd "$PREFILL_NODE"; fi
if [ "${DECODE_KVD:-0}"  = "1" ]; then start_kvd "$DECODE_NODE"; fi

log "=== 3/4 legs ==="
# Launch both legs before waiting on either: they load ~400 GB of weights concurrently, and
# serialising the waits doubles the bring-up for no reason.
on "$PREFILL_NODE" "$COMMON_ENV ROLE=prefill MY_IP=$PREFILL_IP PORT=$PREFILL_PORT \
  DPA=${PREFILL_DPA:-0} MTP=${PREFILL_MTP:-0} KVD=${PREFILL_KVD:-1} \
  bash $KIT_DIR/engine/leg.sh"
on "$DECODE_NODE" "$COMMON_ENV ROLE=decode MY_IP=$DECODE_IP PORT=$DECODE_PORT \
  DPA=${DECODE_DPA:-1} MTP=${DECODE_MTP:-1} KVD=${DECODE_KVD:-0} \
  bash $KIT_DIR/engine/leg.sh"

# Poll /health from INSIDE each node's container. Never curl a PD leg's port from another
# host — and never grep the log for a ready line, because the logs are appended to across
# restarts and a grep matches the PREVIOUS run's line within seconds.
log "waiting for both legs (cold start is minutes — silence is not a hang)"
wait_leg(){
  local h="$1" ip="$2" port="$3" tries="${LEG_TRIES:-120}"
  for i in $(seq 1 "$tries"); do
    if on "$h" "docker exec $CTR curl -sf -m5 http://$ip:$port/health >/dev/null 2>&1"; then
      log "  $h ($ip:$port) serving after $((i * 15))s"; return 0
    fi
    sleep 15
  done
  warn "  $h ($ip:$port) did NOT come up in $((tries * 15))s"; return 1
}
wait_leg "$PREFILL_NODE" "$PREFILL_IP" "$PREFILL_PORT" || die "prefill leg never became ready"
wait_leg "$DECODE_NODE"  "$DECODE_IP"  "$DECODE_PORT"  || die "decode leg never became ready"

log "=== 4/4 router (prefill node) ==="
# The router last: it registers the workers it finds in etcd, and starting it after both legs
# are serving means its first health check already reflects a paired deployment.
on "$PREFILL_NODE" "$COMMON_ENV ROUTER_POLICY=${ROUTER_POLICY:-kv-aware} \
  ROUTER_BACKEND=${ROUTER_BACKEND:-rust} \
  ${KV_PREFILL_W:+KV_PREFILL_W=$KV_PREFILL_W} ${KV_DECODE_W:+KV_DECODE_W=$KV_DECODE_W} \
  bash -c 'source $KIT_DIR/common.sh; start_router $PREFILL_IP'"

log "up. endpoint: http://$PREFILL_IP:$ROUTER_PORT"
log "verify with:  bash cluster/<your-cluster>.sh smoke"
