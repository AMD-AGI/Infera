#!/usr/bin/env bash
# Phase 3b: 1P1D PD-mori with MTP on decode. prefill=chi2832, decode=chi2879(MTP).
set -euo pipefail
IMAGE=rocm/infera:sglang-v0.1.0-rc6
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PREFILL_HOST=chi2832; PREFILL_IP=10.2.122.79
DECODE_HOST=chi2879;  DECODE_IP=10.2.122.10
ETCD_EP=$PREFILL_IP:2379
NATS_PREFILL=nats://$PREFILL_IP:4222
NATS_DECODE=nats://$DECODE_IP:4222
ROUTER_PORT=8100; CONC="${CONC:-64}"
KIT=/mnt/vast/c_huggingface/glm52_p3b
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }
echo "== 1. NATS both nodes =="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker rm -f infera-nats-p3b >/dev/null 2>&1; docker run -d --name infera-nats-p3b --network host nats:2.10 -p 4222 >/dev/null && echo nats-$h"; done
echo "== 2. etcd on prefill =="
J "$PREFILL_HOST" "docker rm -f repro-etcd-p3b >/dev/null 2>&1; docker run -d --name repro-etcd-p3b --network host quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$PREFILL_IP:2379 --listen-client-urls http://0.0.0.0:2379 >/dev/null && echo etcd-up"
sleep 4
echo "== 3. router (http + kv-aware) on prefill =="
J "$PREFILL_HOST" "docker rm -f glm52-router-p3b >/dev/null 2>&1; docker run -d --name glm52-router-p3b --network host -v /mnt/vast:/mnt/vast --entrypoint python3 $IMAGE -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $ETCD_EP --request-transport http --router-policy kv-aware --router-tokenizer-path $MODEL >/dev/null && echo router-up"
echo "== 4. prefill leg (chi2832, no MTP) =="
J "$PREFILL_HOST" "ROLE=prefill DATA_PLANE_IP=$PREFILL_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_PREFILL IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC MTP=0 bash $KIT/engine.sh"
echo "== 5. decode leg (chi2879, MTP=1) =="
J "$DECODE_HOST" "ROLE=decode DATA_PLANE_IP=$DECODE_IP ETCD_EP=$ETCD_EP NATS_SERVER=$NATS_DECODE IMAGE=$IMAGE MODEL=$MODEL SERVED=$SERVED CONC=$CONC MTP=1 bash $KIT/engine.sh"
echo "== legs launching. router :8100 on $PREFILL_IP. =="
