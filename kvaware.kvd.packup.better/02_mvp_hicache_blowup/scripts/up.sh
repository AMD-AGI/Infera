#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 02 — bring up kvd daemon + 1P1D legs on ONE node (chi2879),
# TP4 + TP4, Qwen3-1.7B. Runs INSIDE the container.
#
# Two knobs, both of which round r1 got wrong:
#
#   HICACHE_MODE=ratio|size    passed through to leg.sh (see there)
#   PORT_GAP=<int>             distance between the two legs' --port values.
#                              sglang derives a block of internal ZMQ / nccl
#                              ports from --port. r1's legs were close enough
#                              that decode's derived block overlapped prefill's
#                              and the second leg died on tcp://127.0.0.1:30235.
#                              PORT_GAP=1000 (prefill 30000 / decode 31000) is
#                              the fix that stuck.
#
# NOTE: this does NOT start the router. r1 never had two live legs, so the
# router is out of scope here — the failure is entirely in leg startup.
# ---------------------------------------------------------------------------
set -u
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
MY_IP="${MY_IP:?MY_IP=<data-plane ip of this node>}"
KVD="${KVD:-1}"
KVAWARE="${KVAWARE:-1}"
ETCD_PORT="${ETCD_PORT:-2379}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
TAG="${TAG:-r1}"
HICACHE_MODE="${HICACHE_MODE:-ratio}"
HICACHE_GB="${HICACHE_GB:-16}"
PORT_GAP="${PORT_GAP:-100}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT=$((PREFILL_PORT + PORT_GAP))

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

echo "===== 3. legs (prefill GPU0-3 :$PREFILL_PORT | decode GPU4-7 :$DECODE_PORT, gap=$PORT_GAP) ====="
ROLE=prefill MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=$PREFILL_PORT BASE_GPU=0 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_MODE=$HICACHE_MODE HICACHE_GB=$HICACHE_GB \
  KV_PUB_PORT=5557 LOG=/tmp/$TAG/prefill.log \
  nohup bash /work/leg.sh > /tmp/$TAG/prefill.spawn 2>&1 &
sleep 20
ROLE=decode MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=$DECODE_PORT BASE_GPU=4 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_MODE=$HICACHE_MODE HICACHE_GB=$HICACHE_GB \
  KV_PUB_PORT=5657 LOG=/tmp/$TAG/decode.log \
  nohup bash /work/leg.sh > /tmp/$TAG/decode.spawn 2>&1 &

echo "launched. tag=$TAG hicache_mode=$HICACHE_MODE port_gap=$PORT_GAP  logs in /tmp/$TAG/"
