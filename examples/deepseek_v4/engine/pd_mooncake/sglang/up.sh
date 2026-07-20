#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: bring up sglang PD (mooncake) across nodes by TOPO, via infera.server etcd auto-pairing.
# why : one script for 1p1d/2p1d/2p2d (requirement #6); infera.server auto-pairs all P/D legs
#       from etcd (no static mini-lb list). how: run from a host that can ssh each node; the kit
#       must live on a shared fs path identical on every node (KIT_DIR). caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_env INFERA_IMAGE
require_env PREFILL_NODE; require_env DECODE_NODE; require_env PREFILL_IP; require_env DECODE_IP
require_env KIT_DIR   # shared-fs path to this repro kit, identical on every node (the deepseek_v4 dir)
BACKEND="${BACKEND:-mooncake}"; TOPO="${TOPO:-2p1d}"; CTR="${CTR:-dsv4_pd_sgl}"; CONC="${CONC:-640}"
P2_NODE="${P2_NODE:-}"; P2_IP="${P2_IP:-}"; CASE="engine/pd_mooncake/sglang"
SSH(){ ssh -o StrictHostKeyChecking=no "$@"; }

# what: on a node, start the container + inject host libionic (else RDMA degrades to TCP).
prep_node(){ local h="$1"; log "prep $h"
  SSH "$h" "INFERA_IMAGE=$INFERA_IMAGE INFERA_MODEL_MOUNT=$INFERA_MODEL_MOUNT bash -c 'source $KIT_DIR/common.sh; start_container $CTR'"
  SSH "$h" "bash -s $CTR" <<'IONIC'
CTR="$1"; HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=$(basename "$HL")
docker cp "$HL" "$CTR:/usr/lib/x86_64-linux-gnu/$B"
docker exec "$CTR" bash -lc "cd /usr/lib/x86_64-linux-gnu && ln -sf $B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo active_ports: \$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)"
IONIC
}
# what: launch one leg on a node (engine.sh runs on the HOST; it docker-execs into $CTR itself).
leg(){ local h="$1" role="$2" myip="$3" tp="$4" base="$5" port="$6" chunk="$7"
  SSH "$h" "ROLE=$role MY_IP=$myip ETCD_IP=$PREFILL_IP CTR=$CTR BACKEND=$BACKEND TP=$tp BASE_GPU=$base \
    PORT=$port CONC=$CONC CHUNK=$chunk INFERA_MODEL=$INFERA_MODEL INFERA_IMAGE=$INFERA_IMAGE \
    ETCD_PORT=$ETCD_PORT GID_INDEX=$GID_INDEX bash $KIT_DIR/$CASE/engine.sh"; }

# ---- prep nodes ----
prep_node "$PREFILL_NODE"; prep_node "$DECODE_NODE"
[ "$TOPO" != 1p1d ] && { require_env P2_NODE; require_env P2_IP; prep_node "$P2_NODE"; }

# ---- etcd + router on prefill node (router runs in-container; engine.sh not needed here) ----
SSH "$PREFILL_NODE" "bash -c 'source $KIT_DIR/common.sh; ETCD_PORT=$ETCD_PORT start_etcd repro-etcd $PREFILL_IP'"
SSH "$PREFILL_NODE" "INFERA_MODEL=$INFERA_MODEL INFERA_TOKENIZER=${INFERA_TOKENIZER:-$INFERA_MODEL} ROUTER_PORT=$ROUTER_PORT ETCD_PORT=$ETCD_PORT \
  bash -c 'source $KIT_DIR/common.sh; start_router $CTR $PREFILL_IP'"

# ---- launch legs per TOPO (chunk 163840 = 160k sweet spot; DP-attn auto-on at CONC>=128) ----
case "$TOPO" in
  1p1d) leg "$PREFILL_NODE" prefill "$PREFILL_IP" 8 0 30000 163840
        leg "$DECODE_NODE"  decode  "$DECODE_IP"  8 0 30000 163840 ;;
  2p1d) leg "$PREFILL_NODE" prefill "$PREFILL_IP" 8 0 30000 163840
        leg "$P2_NODE"      prefill "$P2_IP"      8 0 30000 163840
        leg "$DECODE_NODE"  decode  "$DECODE_IP"  8 0 30000 163840 ;;
  2p2d) leg "$PREFILL_NODE" prefill "$PREFILL_IP" 8 0 30000 163840
        leg "$P2_NODE"      prefill "$P2_IP"      8 0 30000 163840
        leg "$DECODE_NODE"  decode  "$DECODE_IP"  4 0 30000 163840
        leg "$DECODE_NODE"  decode  "$DECODE_IP"  4 4 30100 163840 ;;
  *) die "unknown TOPO=$TOPO (want 1p1d|2p1d|2p2d)" ;;
esac

log "sglang PD ($TOPO/$BACKEND) legs launching — cold start ~30min. Poll then smoke:"
log "  KIT_DIR=$KIT_DIR PREFILL_IP=$PREFILL_IP INFERA_MODEL=$INFERA_MODEL bash $DIR/smoke.sh"
