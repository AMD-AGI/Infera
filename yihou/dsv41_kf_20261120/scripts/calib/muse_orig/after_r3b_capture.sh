#!/bin/bash
# v3-180: hold the EP4 queue after r3b, run the Step 2b capture-mode profile, then release the queue.
# Usage (node host): nohup bash after_r3b_capture.sh > $NFS/profiles/calibprof-tp4-c40-v180p3-calib.waiter.log 2>&1 &
set -u
N=/shared_nfs/yihou/muse_agentx_opt_20261027
Q=$N/workspace/coordination/queues/q_ep4rep2.txt
QL=$N/iterations/queue_v3180b.out
ts() { date -u +%FT%TZ; }
for t in infx1106-tp4-nb-c40-r4a infx1106-tp4-nb-ep4perm-c40-r4b; do echo "$t" >> "$Q.done"; done
echo "$(ts) queue held after r3b"
until grep -aq "end infx1106-tp4-nb-c40-r3b" "$QL"; do sleep 60; done
sleep 30
while docker ps --format '{{.Names}}' | grep -q '^yihou-muse-opt-'; do sleep 30; done
echo "$(ts) r3b ended; start capture profile"
cd "$N/calib_profiler/node" && bash step2_profile.sh grant.v3180.json calibprof-tp4-c40-v180p3 40 18 > "$N/profiles/calibprof-tp4-c40-v180p3-calib.driver.log" 2>&1
echo "$(ts) capture profile rc=$?"
sed -i -e '/^infx1106-tp4-nb-c40-r4a$/d' -e '/^infx1106-tp4-nb-ep4perm-c40-r4b$/d' "$Q.done"
echo "$(ts) queue released: $(tr '\n' ' ' < "$Q.done")"
