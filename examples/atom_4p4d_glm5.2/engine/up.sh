#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Bring up the 4P4D deployment on this node:
#   etcd (+ Mooncake Store) -> prefill (GPUs 0-3) + decode (GPUs 4-7) -> router.
# Each run gets a directory under $WORK_DIR/runs that records the settings used;
# smoke.sh, eval.sh, agentx.sh and down.sh write into it.
#
# Usage: bash engine/up.sh            (RUN_ID=<name> to name the run directory)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"

case "$SESSION_AFFINITY" in
    off | prefill | both) ;;
    *) die "SESSION_AFFINITY must be off, prefill or both, not $SESSION_AFFINITY" ;;
esac
case "$ROUTER_BACKEND" in
    python | rust) ;;
    *) die "ROUTER_BACKEND must be python or rust, not $ROUTER_BACKEND" ;;
esac
case "$PREFILL_CPU_TIER" in
    store | lmcache | off) ;;
    *) die "PREFILL_CPU_TIER must be store, lmcache or off, not $PREFILL_CPU_TIER" ;;
esac
# A full decode step, 2 x CONC sequences of MTP_K + 1 tokens, must fit one
# batch; otherwise every step is cut short of max-num-seqs.
(((MTP_K + 1) * 2 * CONC <= DECODE_MAX_BATCHED_TOKENS)) ||
    die "DECODE_MAX_BATCHED_TOKENS=$DECODE_MAX_BATCHED_TOKENS is below (MTP_K + 1) x 2 x CONC = $(((MTP_K + 1) * 2 * CONC))"
for gpus in "$PREFILL_GPUS" "$DECODE_GPUS"; do
    [[ "$gpus" =~ ^[0-9]+(,[0-9]+){3}$ ]] || die "each leg needs exactly four GPUs, got $gpus"
done
# The legs must split the node's eight GPUs between them.
[[ "$(tr , '\n' <<<"$PREFILL_GPUS,$DECODE_GPUS" | sort -u | paste -sd,)" == 0,1,2,3,4,5,6,7 ]] ||
    die "PREFILL_GPUS and DECODE_GPUS must together cover GPUs 0-7 once each"
resolve_model
for c in "${CONTAINERS[@]}"; do
    if docker inspect --type container "$c" >/dev/null 2>&1; then die "$c already exists: bash engine/down.sh first"; fi
done

echo "===== 1. wait for free VRAM ====="
wait_vram_free

RUN_DIR="$WORK_DIR/runs/${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-c$CONC}"
mkdir -p "$RUN_DIR"
for var in "${RUN_VARS[@]}"; do printf '%s=%q\n' "$var" "${!var}"; done >"$RUN_DIR/engine-config.env"
echo "$RUN_DIR" >"$WORK_DIR/current_run"

echo "===== 2. etcd ====="
docker run -d --name "$CTR-etcd" --network host --entrypoint /usr/local/bin/etcd "$ETCD_IMAGE" \
    --advertise-client-urls "http://$NODE_IP:$ETCD_PORT" \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT" >/dev/null
wait_health "http://$NODE_IP:$ETCD_PORT/version" "$CTR-etcd"
# The prefill connects to the Store at startup.
[[ "$PREFILL_CPU_TIER" != store ]] || bash "$KIT/engine/store.sh"

echo "===== 3. prefill and decode ====="
bash "$KIT/engine/worker.sh" prefill
bash "$KIT/engine/worker.sh" decode
# Prefills on other nodes (REMOTE_PREFILLS), started and waited for here.
[[ -z "$REMOTE_PREFILLS" ]] || bash "$REMOTE_PREFILLS" up
# Cold start takes many minutes: weights, AITER JIT, graph capture.
wait_health "http://$NODE_IP:$PREFILL_PORT/health" "$CTR-prefill" "$CTR-decode"
wait_health "http://$NODE_IP:$DECODE_PORT/health" "$CTR-prefill" "$CTR-decode"

echo "===== 4. $ROUTER_BACKEND router (session affinity: $SESSION_AFFINITY) ====="
# ATOM speaks neither NATS requests nor NATS KV events, so the transports are
# explicit. kv-aware needs engine KV events, which this kit does not enable.
read -ra router_extra <<<"$ROUTER_EXTRA_ARGS"
docker run -d --name "$CTR-router" --network host -v "$MODEL:$MODEL:ro" "$IMAGE" \
    python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend "$ROUTER_BACKEND" \
    --discovery-backend etcd --etcd-endpoint "$NODE_IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq \
    --router-policy round-robin --session-affinity "$SESSION_AFFINITY" --router-tokenizer-path "$MODEL" "${router_extra[@]}" >/dev/null
wait_health "$ROUTER_URL/health" "$CTR-router"

echo "===== 5. router sees $PREFILL_NUM_WORKERS prefill and one decode ====="
# A prefill registered as "mixed" would serve requests without PD.
for _ in $(seq 1 12); do
    curl -sf -m 5 "$ROUTER_URL/v1/workers" >"$RUN_DIR/workers.json" || true
    n_prefill="$(grep -o '"disagg_mode": *"prefill"' "$RUN_DIR/workers.json" | wc -l || true)"
    n_decode="$(grep -o '"disagg_mode": *"decode"' "$RUN_DIR/workers.json" | wc -l || true)"
    ((n_prefill == PREFILL_NUM_WORKERS && n_decode == 1)) && break
    sleep 5
done
((n_prefill == PREFILL_NUM_WORKERS && n_decode == 1)) ||
    die "router lists $n_prefill prefill / $n_decode decode workers, want $PREFILL_NUM_WORKERS / 1: $RUN_DIR/workers.json"

echo "===== up: $ROUTER_URL  CONC=$CONC MTP_K=$MTP_K MTP_AL=${MTP_AL:-real} run=$RUN_DIR ====="
