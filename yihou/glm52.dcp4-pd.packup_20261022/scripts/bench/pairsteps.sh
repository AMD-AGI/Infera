#!/bin/bash
# pairsteps.sh TAG...: wavesteps for each server tag at 34K and 150K (bench/results/decodebench.<tag>-{34k,150k}.json).
B=/shared_nfs/yihou_glm52/dcp_pd_20261021/bench
a=()
for t in "$@"; do L=$(ls -t $B/logs/bench-$t-sim.*.log | head -1)
  for c in 34k 150k 200k; do [ -f $B/results/decodebench.$t-$c.json ] && a+=("$t-$c" "$L" "$B/results/decodebench.$t-$c.json"); done
done
python3 $B/wavesteps.py "${a[@]}"
for t in "$@"; do echo "$t errors: $(grep -cE 'Memory access|Traceback' $(ls -t $B/logs/bench-$t-sim.*.log | head -1))"; done
