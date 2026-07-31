#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 03 — bring up kvd daemon + 1P1D legs on ONE node (chi2879),
# TP4 + TP4, Qwen3-1.7B. Runs INSIDE the container.
#
# This is round r2's launcher. The previous round's two fixes are already in:
#   * --hicache-size (absolute GB) instead of the default ratio
#   * legs' --port spaced 1000 apart (30000 / 31000)
#
# What is NOT fixed here is infera/common/net.py:free_tcp_port_block(), which
# is what decides the --kv-events-config base port. Whether the decode leg
# survives depends purely on which net.py is in the container; see run.sh.
#
# The router is NOT started. With only one live leg there is nothing to route.
# ---------------------------------------------------------------------------
set -u
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
MY_IP="${MY_IP:?MY_IP=<data-plane ip of this node>}"
KVD="${KVD:-1}"
KVAWARE="${KVAWARE:-1}"
ETCD_PORT="${ETCD_PORT:-2379}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
TAG="${TAG:-r2}"
HICACHE_GB="${HICACHE_GB:-8}"

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

echo "===== 3. record which net.py is in play (this is the variable under test) ====="
python3 - <<'PY' 2>&1 | tee "/tmp/$TAG/netpy_probe.txt"
from infera.common.net import free_tcp_port_block
bases = [free_tcp_port_block(4) for _ in range(10)]
print("free_tcp_port_block(4) x10 ->", bases)
print("distinct:", len(set(bases)))
print("VERDICT:", "PRE-FIX (deterministic, collision guaranteed)" if len(set(bases)) == 1
      else "POST-FIX (randomised)")
PY

echo "===== 4. legs (prefill GPU0-3 :30000 | decode GPU4-7 :31000) ====="
ROLE=prefill MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=30000 BASE_GPU=0 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
  KV_PUB_PORT=5557 LOG=/tmp/$TAG/prefill.log \
  nohup bash /work/leg.sh > /tmp/$TAG/prefill.spawn 2>&1 &
sleep 20
ROLE=decode MY_IP=$MY_IP ETCD_IP=$MY_IP MODEL=$MODEL PORT=31000 BASE_GPU=4 TP=4 \
  KVD=$KVD KVAWARE=$KVAWARE KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
  KV_PUB_PORT=5657 LOG=/tmp/$TAG/decode.log \
  nohup bash /work/leg.sh > /tmp/$TAG/decode.spawn 2>&1 &

echo "launched. tag=$TAG  logs in /tmp/$TAG/"
