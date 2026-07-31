#!/bin/bash
# Bring up: kvd daemon + etcd + infera router (kv-aware) + 1P1D legs (DP-attention),
# all on ONE node (chi2879), TP4+TP4. Model = Qwen3-1.7B (cheap MVP; the point is to
# validate the kvaware/kvd wiring, not the model).
# Runs INSIDE the container.
set -u
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
MY_IP="${MY_IP:?}"
KVD="${KVD:-1}"
KVAWARE="${KVAWARE:-1}"
POLICY="${POLICY:-kv-aware}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
ETCD_PORT="${ETCD_PORT:-2379}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
TAG="${TAG:-run}"

mkdir -p /tmp/kvd /tmp/kvd-long "/tmp/$TAG"

echo "===== 0. clean ====="
pkill -9 -f "infera.engine.sglang" 2>/dev/null; pkill -9 -f "infera.server" 2>/dev/null
pkill -9 -f "infera.kvd" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang::" 2>/dev/null; sleep 5
rm -f "$KVD_SOCK"

if [ "$KVD" = "1" ]; then
  echo "===== 1. kvd daemon ====="
  # RAM tier deliberately small so the working set spills to the L3 disk tier.
  nohup python3 -m infera.kvd --socket "$KVD_SOCK" --max-bytes 8G \
      --long-path /tmp/kvd-long --long-bytes 64G --log-level INFO \
      > "/tmp/$TAG/kvd.log" 2>&1 &
  for i in $(seq 1 30); do [ -S "$KVD_SOCK" ] && break; sleep 1; done
  [ -S "$KVD_SOCK" ] || { echo "FATAL kvd socket never appeared"; tail -30 "/tmp/$TAG/kvd.log"; exit 1; }
  echo "kvd up: $KVD_SOCK"
fi

echo "===== 2. etcd ====="
if ! curl -sf -m3 "http://$MY_IP:$ETCD_PORT/version" >/dev/null 2>&1; then
  echo "NOTE: no etcd at $MY_IP:$ETCD_PORT — start it on the HOST first."; exit 1
fi
echo "etcd reachable"

echo "===== 3. legs (prefill GPU0-3 :30000 | decode GPU4-7 :31000) ====="
# Ports spaced 1000 apart: sglang derives a block of internal ZMQ ports from --port,
# and adjacent ports made the decode leg collide with prefill's (round 1: ZMQError
# 'Address already in use' on tcp://127.0.0.1:30235).
ROLE=prefill MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=30000 BASE_GPU=0 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=${HICACHE_GB:-8} KV_PUB_PORT=5557 LOG=/tmp/$TAG/prefill.log \
  nohup bash /work/leg.sh > /tmp/$TAG/prefill.spawn 2>&1 &
sleep 20
ROLE=decode MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=31000 BASE_GPU=4 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=${HICACHE_GB:-8} KV_PUB_PORT=5657 LOG=/tmp/$TAG/decode.log \
  nohup bash /work/leg.sh > /tmp/$TAG/decode.spawn 2>&1 &

echo "===== 4. router (policy=$POLICY) ====="
sleep 5
nohup python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
  --discovery-backend etcd --etcd-endpoint "$MY_IP:$ETCD_PORT" \
  --request-transport http --kv-event-transport zmq \
  --router-policy "$POLICY" --router-tokenizer-path "$MODEL" \
  > "/tmp/$TAG/router.log" 2>&1 &

echo "launched. tag=$TAG  logs in /tmp/$TAG/"
