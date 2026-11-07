#!/usr/bin/env bash
# n02-25 GPU 1 segfaults in its first kernel (11-07 01:07, 3/3): there this item goes back to the queue for the other holder,
# drops its running/ entry, and parks this holder until its job ends.
if [[ "$(hostname -s)" == smci355-ccs-aus-n02-25 ]]; then
    H=/apps/liyingli/holder1 n="$(basename "$0")"
    cp "$0" "$H/queue/"; rm -f "$H/running/$n" "$H/running/${n%.sh}.job"
    echo "$(date -u +%T) v2 ${n%.sh}: not on n02-25 (GPU 1 bad); back to the queue, this holder parks" >> /perf_apps/liyingli/vfix-logs/v2/v2.out
    exec sleep 100000
fi
# v2 kit Pareto, one-node 4+4 (MC_DISABLE_HIP=1): DP balance 400k with K5 at C32 (DP balance moved C64 and C104 up and right:
# C64 K5 42,284 / p90 74.0 -> 43,245 / 81.6; C104 K3 52,409 / 60.6 -> 53,460 / 65.4).
MTP_STEPS=5 DECODE_REQUEST_WEIGHT=400000 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-dpbal-k5-c32 sweep 1node 32
