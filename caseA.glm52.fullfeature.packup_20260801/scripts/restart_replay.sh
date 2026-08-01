#!/usr/bin/env bash
# The check that ATTRIBUTES reuse to kvd rather than to the in-GPU radix cache:
# restart the prefill engine (which empties that cache) while the kvd daemon and
# its L3 keep running, then replay the SAME prefixes. Hits must climb with NO new
# sets -- reuse that could only have come from L3.
#
# A latency win proves nothing here: sglang's radix cache serves a repeated
# prefix without ever touching L3.
#
# Bench-kit version: bench_run container, the ctx=262144 frozen config, and the
# leg script re-copied on launch (the shared-fs copy drifts from the container's).
set -u
CTR="${CTR:-bench_run}"
MY_IP="${MY_IP:-10.2.122.10}"
MTP="${MTP:-0}"
TAG="${TAG:-p2}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
W=/mnt/vast/c_huggingface/bench_20260801
LOG="$W/logs/${TAG}_prefill_restart.log"

echo "=== kvd BEFORE restart ==="
docker exec "$CTR" python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

echo "=== restarting PREFILL engine only (kvd daemon stays up) ==="
docker cp "$W/scripts/glm52_leg.sh" "$CTR":/glm52_leg.sh >/dev/null
docker exec "$CTR" bash -c "
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f infera.engine.sglang 2>/dev/null
  for i in \$(seq 1 20); do
    n=\$(ps aux | grep -E 'launch_server|infera.engine' | grep -v grep | wc -l)
    [ \"\$n\" -eq 0 ] && break
    sleep 2
  done"
docker exec -d "$CTR" env ROLE=prefill MY_IP="$MY_IP" ETCD_IP="$MY_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 \
  CTX=262144 ISL=8192 TP=8 DPA=1 CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \
  KVAWARE=1 KVD=1 HICACHE_GB=16 KVD_SOCK=/tmp/kvd/kvd.sock MTP="$MTP" \
  LOG="$LOG" bash /glm52_leg.sh

echo "  waiting for prefill ready..."
# NOT a log grep: this appends to $LOG, so a pre-restart "ready to roll" matches
# within seconds and the replay runs against an engine still loading weights.
ok=0
for i in $(seq 1 90); do
  if docker exec "$CTR" curl -sf -m5 "http://$MY_IP:30000/health" >/dev/null 2>&1; then
    echo "  prefill serving after $((i * 10))s"; ok=1; break
  fi
  sleep 10
done
[ "$ok" -eq 1 ] || { echo "  PREFILL DID NOT COME BACK" >&2; tail -20 "$LOG"; exit 1; }
sleep 15
echo "=== done; now replay prefix_reuse.py and re-read statctl ==="
