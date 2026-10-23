#!/usr/bin/env bash
# queue_ppda2 (task ppdc 2026-10-23, ON 135; user OK): after orch16 ppda1 is done, run orch16 ppda2 80 80 (A1 only).
D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
for i in $(seq 1 600); do grep -q "orch16 done" $D/orch16.ppda1.yihou.log && break; sleep 30; done
grep -q "orch16 done" $D/orch16.ppda1.yihou.log || { echo "ppda1 orch not done after 5 h"; exit 1; }
bash $D/orch16.yihou.sh ppda2 80 80
