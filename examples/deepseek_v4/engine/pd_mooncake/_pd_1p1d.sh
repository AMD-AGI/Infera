#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: shared 1P1D bring-up for atom/vllm PD (prep 2 nodes, etcd+router on prefill, 2 legs).
# why : atom & vllm PD are 1p1d-only; this removes per-engine boilerplate. how: sourced by their
#       up.sh which sets CTR + CASE then calls pd_1p1d_up; kit must live on shared fs (KIT_DIR).
set -euo pipefail

# what: full 1P1D bring-up. arg $1 = case subpath (e.g. engine/pd_mooncake/atom). caller = up.sh.
# why : one place for container->ionic->etcd->router->legs, all over the shared-fs kit.
pd_1p1d_up(){
  local CASE="$1"; local DIR; DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  source "$DIR/../../../common.sh"
  require_env INFERA_MODEL; require_env INFERA_IMAGE; require_env KIT_DIR
  require_env PREFILL_NODE; require_env DECODE_NODE; require_env PREFILL_IP; require_env DECODE_IP
  local SSH="ssh -o StrictHostKeyChecking=no"
  for pair in "$PREFILL_NODE" "$DECODE_NODE"; do local h="$pair"; log "prep $h"
    $SSH "$h" "INFERA_IMAGE=$INFERA_IMAGE INFERA_MODEL_MOUNT=$INFERA_MODEL_MOUNT bash -c 'source $KIT_DIR/common.sh; start_container $CTR'"
    $SSH "$h" "bash -s $CTR" <<'IONIC'
CTR="$1"; HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=$(basename "$HL")
docker cp "$HL" "$CTR:/usr/lib/x86_64-linux-gnu/$B"
docker exec "$CTR" bash -lc "cd /usr/lib/x86_64-linux-gnu && ln -sf $B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo active_ports: \$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)"
IONIC
  done
  $SSH "$PREFILL_NODE" "bash -c 'source $KIT_DIR/common.sh; ETCD_PORT=$ETCD_PORT start_etcd repro-etcd $PREFILL_IP'"
  $SSH "$PREFILL_NODE" "INFERA_MODEL=$INFERA_MODEL INFERA_TOKENIZER=${INFERA_TOKENIZER:-$INFERA_MODEL} ROUTER_PORT=$ROUTER_PORT ETCD_PORT=$ETCD_PORT \
    bash -c 'source $KIT_DIR/common.sh; start_router $CTR $PREFILL_IP'"
  local ENV="CTR=$CTR ETCD_IP=$PREFILL_IP ETCD_PORT=$ETCD_PORT GID_INDEX=$GID_INDEX INFERA_MODEL=$INFERA_MODEL INFERA_IMAGE=$INFERA_IMAGE"
  $SSH "$PREFILL_NODE" "$ENV ROLE=prefill MY_IP=$PREFILL_IP bash $KIT_DIR/$CASE/engine.sh"
  $SSH "$DECODE_NODE"  "$ENV ROLE=decode  MY_IP=$DECODE_IP  bash $KIT_DIR/$CASE/engine.sh"
  log "1P1D legs launching — cold start ~30min. Poll then: KIT_DIR=$KIT_DIR PREFILL_IP=$PREFILL_IP INFERA_MODEL=$INFERA_MODEL bash $DIR/smoke.sh"
}
