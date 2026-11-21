#!/bin/bash
# Replaces after_r3b_capture.sh (its end-of-r3b grep used a wrong tag): run the v180p3 capture profile now and write
# the same completion line the p4 waiter watches for.
set -u
N=/shared_nfs/yihou/muse_agentx_opt_20261027
W3=$N/profiles/calibprof-tp4-c40-v180p3-calib.waiter.log
ts() { date -u +%FT%TZ; }
while docker ps --format '{{.Names}}' | grep -q '^yihou-muse-opt-'; do sleep 20; done
echo "$(ts) start capture profile (run_p3_now)" >> "$W3"
cd "$N/calib_profiler/node" && bash step2_profile.sh grant.v3180.json calibprof-tp4-c40-v180p3 40 18 > "$N/profiles/calibprof-tp4-c40-v180p3-calib.driver.log" 2>&1
echo "$(ts) capture profile rc=$?" >> "$W3"
