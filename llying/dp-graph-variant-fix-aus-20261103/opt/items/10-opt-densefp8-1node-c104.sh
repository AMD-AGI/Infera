#!/usr/bin/env bash
# One-node A/B for dense FP8: arm-C-prod-ptpcfix-densefp8 (with dense-fp8/tune's FP8 GEMM rows when present) at 4+4
# C104 on the same node as 09.
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item4.sh densefp8-1node 100 score arm-C-prod-ptpcfix-densefp8.env 104
