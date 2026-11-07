#!/usr/bin/env bash
# v2 kit Pareto, one-node 4+4 (MC_DISABLE_HIP=1): DP balance 400k with K3 at C88 (DP balance moved C64 and C104 up and right:
# C64 K5 42,284 / p90 74.0 -> 43,245 / 81.6; C104 K3 52,409 / 60.6 -> 53,460 / 65.4).
MTP_STEPS=3 DECODE_REQUEST_WEIGHT=400000 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-dpbal-k3-c88 sweep 1node 88
