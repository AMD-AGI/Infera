#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Bring up the deployment from the prefill node: etcd -> two prefills here and
# the decode on the decode node -> router. Each run gets a directory under
# $WORK_DIR/runs that records the settings used; smoke.sh, eval.sh, agentx.sh
# and down.sh write into it.
#
# Usage: bash engine/up.sh            (RUN_ID=<name> to name the run directory)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
[[ -d "$MODEL" ]] || die "MODEL is not a directory on this host: $MODEL"

RUN_DIR="$WORK_DIR/runs/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
for var in "${KIT_VARS[@]}"; do
    printf '%s=%q\n' "$var" "${!var}"
done >"$RUN_DIR/engine-config.env"
echo "$RUN_DIR" >"$WORK_DIR/current_run"

echo "===== 1. etcd ====="
docker run -d --name "$CTR-etcd" --network host --entrypoint /usr/local/bin/etcd "$ETCD_IMAGE" \
    --advertise-client-urls "http://$PREFILL_NODE_IP:$ETCD_PORT" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT" >/dev/null
wait_health "http://$PREFILL_NODE_IP:$ETCD_PORT/version" "$CTR-etcd"

echo "===== 2. engines: 2 prefills here, decode on $DECODE_SSH (cold start takes 10-15 minutes) ====="
bash "$KIT/engine/worker.sh" prefill 0
bash "$KIT/engine/worker.sh" prefill 1
# The decode node reads the same env.sh; this invocation's settings ride along.
forward="$(sed 's/$/ /' "$RUN_DIR/engine-config.env" | tr -d '\n')"
on_decode "$forward bash $(printf %q "$KIT/engine/worker.sh") decode"
wait_health "http://$PREFILL_NODE_IP:$PREFILL_PORT/health" "$CTR-prefill0"
wait_health "http://$PREFILL_NODE_IP:$((PREFILL_PORT + 256))/health" "$CTR-prefill1"
wait_health "http://$DECODE_NODE_IP:$DECODE_PORT/health" "$DECODE_CONTAINER" decode

echo "===== 3. router: kv-aware, session affinity on prefill picks ====="
read -ra router_extra <<<"$ROUTER_EXTRA_ARGS"
docker run -d --name "$CTR-router" --network host -v "$MODEL:$MODEL:ro" "$IMAGE" \
    python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend rust \
    --discovery-backend etcd --etcd-endpoint "$PREFILL_NODE_IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" \
    --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 \
    --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on \
    "${router_extra[@]}" >/dev/null
wait_health "$ROUTER_URL/health" "$CTR-router"
sleep 10
curl -sf -m 10 "$ROUTER_URL/v1/workers" >"$RUN_DIR/workers.json" || true
echo "router workers (expect 2 prefill + 1 decode): $(cat "$RUN_DIR/workers.json")"
echo "===== up: $ROUTER_URL  SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN:-real}  run=$RUN_DIR ====="
