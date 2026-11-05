#!/usr/bin/env bash
# MTP K crossover, K5 half: one-node 4+4 C64, MC_DISABLE_HIP=1, K5 (3.61), 3600 s; against 19 (K3 C64 41,557) per
# busiest-rank load.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh k5-nohip-1node-c64 105 score arm-C-prod-ptpcfix-nohip-k5.env 64
