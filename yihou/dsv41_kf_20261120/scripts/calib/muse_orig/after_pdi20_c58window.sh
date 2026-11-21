#!/bin/bash
# v3-039: after ep-c58-pdi20 ends, run one calibrated window at C58 (best stack) with the fork json-tool.
set -u
N=/shared_nfs/yihou/muse_agentx_opt_20261027
QL=$N/iterations/queue_v3121.out
ts() { date -u +%FT%TZ; }
until grep -aq "end infx1106-tp4-ep-c58-pdi20-r2" "$QL"; do sleep 60; done
sleep 30
while docker ps --format '{{.Names}}' | grep -q '^yihou-muse-opt-'; do sleep 20; done
echo "$(ts) start C58 window"
cd "$N/calib_profiler/node" && CALIB_SCORE_S=900 CALIB_WB_S=900 CALIB_SB_S=1800 bash step2_profile.sh grant.v3121.json calibprof-tp4-c58-ep4-v121p1 58 30 > "$N/profiles/calibprof-tp4-c58-ep4-v121p1-calib.driver.log" 2>&1
echo "$(ts) C58 window rc=$?"
