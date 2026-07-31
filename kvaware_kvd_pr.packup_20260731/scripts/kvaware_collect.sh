#!/bin/bash
set -u
CTR=kvaware_kvd_final
OUT=/mnt/vast/c_huggingface/kvaware_kvd_final/evidence
mkdir -p "$OUT"
docker cp $CTR:/tmp/final_results "$OUT/" 2>/dev/null
cp /root/tests.log "$OUT/tests_full.log" 2>/dev/null
cp /root/readtest.log "$OUT/kvd_reuse_before_restart.log" 2>/dev/null
cp /root/replay.log "$OUT/kvd_reuse_after_restart.log" 2>/dev/null
cp /root/restart.log "$OUT/restart.log" 2>/dev/null
docker exec $CTR bash -c 'cat /tmp/reuse.py' > "$OUT/reuse.py" 2>/dev/null
docker logs ${CTR}_etcd > "$OUT/etcd.log" 2>&1 || true
docker exec $CTR bash -c 'cat /tmp/router.log' > "$OUT/router.log" 2>/dev/null
docker exec $CTR bash -c 'cat /tmp/kvd.log' > "$OUT/kvd_daemon_prefill.log" 2>/dev/null
echo "== collected =="
ls -la "$OUT" | tail -12
du -sh "$OUT"
