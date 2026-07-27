#!/usr/bin/env bash
# Phase 2 1P1D PD-mooncake bring-up. MODE=tcp|rdma (default tcp).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/pd_env.sh"
JUMP="root@149.28.124.225"; CONC="${CONC:-64}"; MODE="${MODE:-tcp}"
KIT=/mnt/vast/c_huggingface/glm52_p2
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$JUMP" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }
echo "== 1. NATS on both nodes =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker rm -f infera-nats-p2 >/dev/null 2>&1; docker run -d --name infera-nats-p2 --network host nats:2.10 -p 4222 >/dev/null && echo nats-$h"; done
echo "== 2. etcd on prefill =="
J "$PREFILL_HOST" "docker rm -f repro-etcd-p2 >/dev/null 2>&1; docker run -d --name repro-etcd-p2 --network host quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$PREFILL_IP:2379 --listen-client-urls http://0.0.0.0:2379 >/dev/null && echo etcd-up"
sleep 4
echo "== 3. router (http transport + kv-aware) on prefill =="
J "$PREFILL_HOST" "docker rm -f glm52-router-p2 >/dev/null 2>&1; docker run -d --name glm52-router-p2 --network host -v /mnt/vast:/mnt/vast --entrypoint python3 $IMAGE -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $ETCD_EP --request-transport http --router-policy kv-aware --router-tokenizer-path $MODEL >/dev/null && echo router-up"
echo "== 4. prefill leg ($PREFILL_HOST) MODE=$MODE =="
J "$PREFILL_HOST" "ROLE=prefill DATA_PLANE_IP=$PREFILL_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_PREFILL IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC MODE=$MODE bash $KIT/engine.sh"
echo "== 5. decode leg ($DECODE_HOST) MODE=$MODE =="
J "$DECODE_HOST" "ROLE=decode DATA_PLANE_IP=$DECODE_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_DECODE IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC MODE=$MODE bash $KIT/engine.sh"
echo "== legs launching (MODE=$MODE). Poll :$ROUTER_PORT/health then probe. =="
