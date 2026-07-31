#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 04 — kvd daemon + 1P1D legs + infera router on ONE node (chi2879),
# TP4 + TP4, Qwen3-1.7B. Runs INSIDE the container.
#
# This is the first round where BOTH legs come up, so it is also the first
# round that starts the ROUTER and can actually take a request. Which is what
# makes the same-host RDMA failure visible.
#
#   FORCE_TCP=0 -> real mooncake RDMA. Expect HTTP 500 on every completion.
#   FORCE_TCP=1 -> MC_FORCE_TCP=1 workaround. Expect completions to succeed.
#
# All three fixes from the earlier rounds are in:
#   --hicache-size 8, ports 1000 apart, patched net.py (applied by run.sh),
#   and per-leg --kv-events-bind (5557 / 5657).
# ---------------------------------------------------------------------------
set -u
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
MY_IP="${MY_IP:?MY_IP=<data-plane ip of this node>}"
KVD="${KVD:-1}"
KVAWARE="${KVAWARE:-1}"
POLICY="${POLICY:-kv-aware}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
ETCD_PORT="${ETCD_PORT:-2379}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
TAG="${TAG:-r3}"
HICACHE_GB="${HICACHE_GB:-8}"
FORCE_TCP="${FORCE_TCP:-0}"

mkdir -p /tmp/kvd /tmp/kvd-long "/tmp/$TAG"

echo "===== 0. clean ====="
pkill -9 -f "infera.engine.sglang" 2>/dev/null
pkill -9 -f "infera.server"        2>/dev/null
pkill -9 -f "infera.kvd"           2>/dev/null
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang::"             2>/dev/null
sleep 5
rm -f "$KVD_SOCK"

if [ "$KVD" = "1" ]; then
  echo "===== 1. kvd daemon ====="
  nohup python3 -m infera.kvd --socket "$KVD_SOCK" --max-bytes 8G \
      --long-path /tmp/kvd-long --long-bytes 64G --log-level INFO \
      > "/tmp/$TAG/kvd.log" 2>&1 &
  for _ in $(seq 1 30); do [ -S "$KVD_SOCK" ] && break; sleep 1; done
  [ -S "$KVD_SOCK" ] || { echo "FATAL kvd socket never appeared"; tail -30 "/tmp/$TAG/kvd.log"; exit 1; }
  echo "kvd up: $KVD_SOCK"
fi

echo "===== 2. etcd ====="
if ! curl -sf -m3 "http://$MY_IP:$ETCD_PORT/version" >/dev/null 2>&1; then
  echo "NOTE: no etcd at $MY_IP:$ETCD_PORT — start it on the HOST first."; exit 1
fi
echo "etcd reachable"

echo "===== 3. RDMA rails visible in this container ====="
# If this is not 8, host libionic was not injected and the whole transport
# question is moot — RDMA has already silently degraded.
echo "active_ports=$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)"

echo "===== 4. legs (prefill GPU0-3 :30000 | decode GPU4-7 :31000) force_tcp=$FORCE_TCP ====="
ROLE=prefill MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=30000 BASE_GPU=0 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
  FORCE_TCP=$FORCE_TCP KV_PUB_PORT=5557 LOG=/tmp/$TAG/prefill.log \
  nohup bash /work/leg.sh > /tmp/$TAG/prefill.spawn 2>&1 &
sleep 20
ROLE=decode MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=31000 BASE_GPU=4 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
  FORCE_TCP=$FORCE_TCP KV_PUB_PORT=5657 LOG=/tmp/$TAG/decode.log \
  nohup bash /work/leg.sh > /tmp/$TAG/decode.spawn 2>&1 &

echo "===== 5. router (policy=$POLICY) ====="
sleep 5
nohup python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
  --discovery-backend etcd --etcd-endpoint "$MY_IP:$ETCD_PORT" \
  --request-transport http --kv-event-transport zmq \
  --router-policy "$POLICY" --router-tokenizer-path "$MODEL" \
  > "/tmp/$TAG/router.log" 2>&1 &

echo "launched. tag=$TAG force_tcp=$FORCE_TCP  logs in /tmp/$TAG/"
