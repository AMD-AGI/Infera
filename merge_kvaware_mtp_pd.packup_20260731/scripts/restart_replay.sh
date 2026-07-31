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
docker exec "$CTR" bash -c "pkill -9 -f 'infera.engine.sglang'; sleep 8"
docker exec -d "$CTR" env ROLE=prefill MY_IP="$MY_IP" ETCD_IP="$MY_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 \
  KVD_SOCK=/tmp/kvd/kvd.sock MTP="$MTP" LOG="$LOG" bash /glm52_leg.sh

echo "  waiting for prefill ready..."
for i in $(seq 1 90); do
  grep -aq "ready to roll" "$LOG" 2>/dev/null && { echo "  ready after ${i}0s"; break; }
  sleep 10
done
grep -aq "ready to roll" "$LOG" || { echo "  PREFILL DID NOT COME BACK" >&2; tail -20 "$LOG"; exit 1; }
sleep 15
