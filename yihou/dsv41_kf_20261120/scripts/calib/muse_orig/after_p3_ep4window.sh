#!/bin/bash
# v3-180: after the v180p3 capture profile, run one EP4-perm window profile (v180p4) if it can finish before the lease.
set -u
N=/shared_nfs/yihou/muse_agentx_opt_20261027
Q=$N/workspace/coordination/queues/q_ep4rep2.txt
W3=$N/profiles/calibprof-tp4-c40-v180p3-calib.waiter.log
ts() { date -u +%FT%TZ; }
until grep -aq "capture profile rc=" "$W3"; do sleep 30; done
for t in infx1106-tp4-nb-c40-r4a infx1106-tp4-nb-ep4perm-c40-r4b; do grep -qx "$t" "$Q.done" || echo "$t" >> "$Q.done"; done
echo "$(ts) p3 done; queue re-held"
latest=$(date -d 2026-11-13T12:58:00Z +%s)
[ "$(date +%s)" -lt "$latest" ] || { echo "$(ts) too late for p4 (needs ~45 min before stop_by 13:46Z)"; exit 0; }
while docker ps --format '{{.Names}}' | grep -q '^yihou-muse-opt-'; do sleep 20; done
cd "$N/calib_profiler/node" && bash step2_profile.sh grant.v3180.json calibprof-tp4-c40-ep4-v180p4 40 18 > "$N/profiles/calibprof-tp4-c40-ep4-v180p4-calib.driver.log" 2>&1
echo "$(ts) p4 rc=$?"
