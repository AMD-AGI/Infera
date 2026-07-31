#!/bin/bash
# The check that ATTRIBUTES reuse to kvd: restart the engine legs (which empties
# the in-GPU radix cache) and replay the SAME prefixes. The kvd daemon keeps
# running, so its L3 survives. Hits should climb with no new sets.
set -u
CTR=kvaware_kvd_final
KIT=/mnt/vast/c_huggingface/kvaware_kvd_final
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
echo "=== kvd BEFORE restart ==="
docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
echo "=== restarting PREFILL engine only (kvd daemon stays up) ==="
docker exec $CTR bash -c "pkill -9 -f 'infera.engine.sglang'; sleep 8"
docker exec -d $CTR env ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MODEL=$MODEL \
  SERVED=glm5.2-mxfp4 PORT=30000 KVAWARE=1 KVD=1 HICACHE_GB=16 KVD_SOCK=/tmp/kvd/kvd.sock \
  LOG=$KIT/prefill_restart.log bash /glm52_leg.sh
echo "  prefill restarting; waiting for ready..."
for i in $(seq 1 60); do
  grep -q "ready to roll" "$KIT/prefill_restart.log" 2>/dev/null && { echo "  ready after ${i}0s"; break; }
  sleep 10
done
