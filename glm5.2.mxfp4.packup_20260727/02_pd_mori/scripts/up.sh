#!/usr/bin/env bash
# Phase 3 1P1D PD-mori bring-up orchestrator. Run from a host that can ssh both nodes.
# Order: NATS(both) + etcd(prefill) + router(prefill) -> prefill leg (chi2878) -> decode leg (chi2879).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/pd_env.sh"
JUMP="root@149.28.124.225"
CONC="${CONC:-64}"
KIT=/mnt/vast/c_huggingface/glm52_p3   # shared-fs copy of engine.sh
SSH(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$@"; }
J(){ SSH "$JUMP" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }

echo "== 1. NATS broker on both engine nodes =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker rm -f infera-nats-p3 >/dev/null 2>&1; docker run -d --name infera-nats-p3 --network host nats:2.10 -p 4222 >/dev/null && echo nats-up-$h"
done

echo "== 2. etcd on prefill node ($PREFILL_HOST) =="
J "$PREFILL_HOST" "docker rm -f repro-etcd-p3 >/dev/null 2>&1; docker run -d --name repro-etcd-p3 --network host quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$PREFILL_IP:2379 --listen-client-urls http://0.0.0.0:2379 >/dev/null && echo etcd-up"
sleep 4

echo "== 3. infera.server router on prefill node (kv-aware, :$ROUTER_PORT) =="
# router runs in a lightweight exec into a throwaway container of the image (needs infera pkg).
J "$PREFILL_HOST" "docker rm -f glm52-router-p3 >/dev/null 2>&1; docker run -d --name glm52-router-p3 --network host -v /mnt/vast:/mnt/vast --entrypoint python3 $IMAGE -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $ETCD_EP --request-transport nats --nats-server $NATS_PREFILL --router-policy round-robin --router-tokenizer-path $MODEL >/dev/null && echo router-up"

echo "== 4. prefill leg on $PREFILL_HOST =="
J "$PREFILL_HOST" "ROLE=prefill DATA_PLANE_IP=$PREFILL_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_PREFILL IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC bash $KIT/engine.sh"

echo "== 5. decode leg on $DECODE_HOST =="
J "$DECODE_HOST" "ROLE=decode DATA_PLANE_IP=$DECODE_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_DECODE IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC bash $KIT/engine.sh"

echo "== legs launching — cold start ~5-30min. Poll router :$ROUTER_PORT/health, then probe. =="
