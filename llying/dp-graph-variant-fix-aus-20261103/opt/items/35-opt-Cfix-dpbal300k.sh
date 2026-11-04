#!/usr/bin/env bash
# DP balance at weight 300,000 on the best correct config, C104 (vs 53,666 at weight 0).
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh Cfix-dpbal300k 100 score arm-C-prod-ptpcfix-dpbal300k.env 104
