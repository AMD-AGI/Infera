#!/usr/bin/env bash
# Round 3 (slot0 = main, slot1 = pr, as round 1): round 1's throughput phase failed on a wrong module name.
# Starts when the first driver has finished.
set -uo pipefail
AB=/perf_apps/liyingli/sgl48302-ab-20261118
cd "$AB"
while pgrep -f "bash driver.sh" > /dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) driver_r3 start" >> summary.txt
bash ab_node.sh smci355-ccs-aus-n04-33 P 3 main pr > runs/r3-P.driver.log 2>&1 &
a=$!
bash ab_node.sh smci355-ccs-aus-n06-25 E 3 main pr > runs/r3-E.driver.log 2>&1 &
b=$!
wait $a $b
echo "$(date -u +%FT%TZ) driver_r3 done" >> summary.txt
