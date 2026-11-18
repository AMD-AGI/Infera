#!/usr/bin/env bash
# sglang#48302 A/B driver (login node, inside job 36659's two nodes):
#   n04-33: cfg P (image defaults), n06-25: cfg E (E11 collectives); each node runs main and pr side by side.
#   round 1: slot0 = main, slot1 = pr; round 2: swapped, so the GPU / NUMA slot cancels out.
# Unit tests (CPU) run next to round 1.
set -uo pipefail
AB=/perf_apps/liyingli/sgl48302-ab-20261118
N1=smci355-ccs-aus-n04-33 N2=smci355-ccs-aus-n06-25
cd "$AB" && mkdir -p runs
echo "$(date -u +%FT%TZ) driver start" >> summary.txt
bash unit.sh $N2 > runs/unit.driver.log 2>&1 &
for r in "1 main pr" "2 pr main"; do
    set -- $r
    bash ab_node.sh $N1 P "$1" "$2" "$3" > "runs/r$1-P.driver.log" 2>&1 &
    a=$!
    bash ab_node.sh $N2 E "$1" "$2" "$3" > "runs/r$1-E.driver.log" 2>&1 &
    b=$!
    wait $a $b
done
wait
echo "$(date -u +%FT%TZ) driver done" >> summary.txt
