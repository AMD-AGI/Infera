#!/usr/bin/env bash
# The check that ATTRIBUTES reuse to kvd rather than to the in-GPU radix cache:
# restart the prefill engine (which empties that cache) while the kvd daemon and
# its L3 keep running, then replay the SAME prefixes. Hits must climb with NO new
# sets -- reuse that could only have come from L3.
set -u
CTR="${CTR:-merge_g0}"
MY_IP="${MY_IP:-10.2.122.10}"
MTP="${MTP:-0}"
TAG="${TAG:-g0}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
KIT=/mnt/vast/c_huggingface/merge_20260731
LOG="$KIT/logs/${TAG}_prefill_restart.log"

echo "=== kvd BEFORE restart ==="
docker exec "$CTR" python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

echo "=== restarting PREFILL engine only (kvd daemon stays up) ==="
# Kill the launch_server CHILD too, not just the infera wrapper: the child holds
# the DP kv-event port block, and the relaunch then dies with "port_base at N is
# not available".
docker exec "$CTR" bash -c "
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f infera.engine.sglang 2>/dev/null
  for i in \$(seq 1 20); do
    n=\$(ps aux | grep -E 'launch_server|infera.engine' | grep -v grep | wc -l)
    [ \"\$n\" -eq 0 ] && break
    sleep 2
  done"
docker exec -d "$CTR" env ROLE=prefill MY_IP="$MY_IP" ETCD_IP="$MY_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 \
  KVD_SOCK=/tmp/kvd/kvd.sock MTP="$MTP" LOG="$LOG" bash /glm52_leg.sh

echo "  waiting for prefill ready..."
# NOT a log grep. This script APPENDS to $LOG, so a "ready to roll" from the
# pre-restart run matches within seconds and the replay below then runs against
# an engine still loading weights -- the kvd counters it reads would be
# meaningless. Observed on the built-image run as "ready after 10s" for
# something that takes minutes. Poll the endpoint, which can only answer once
# the engine is actually up.
ok=0
for i in $(seq 1 90); do
  if docker exec "$CTR" curl -sf -m5 "http://$MY_IP:30000/health" >/dev/null 2>&1; then
    echo "  prefill serving after $((i * 10))s"; ok=1; break
  fi
  sleep 10
done
[ "$ok" -eq 1 ] || { echo "  PREFILL DID NOT COME BACK" >&2; tail -20 "$LOG"; exit 1; }
sleep 15
