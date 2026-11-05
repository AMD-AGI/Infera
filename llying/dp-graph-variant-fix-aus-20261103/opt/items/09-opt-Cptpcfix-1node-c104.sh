#!/usr/bin/env bash
# One-node A/B for dense FP8, reference half: C + PTPC/#45800 (pkg kit) at 4+4 C104 on the one-node holder (prefill
# GPUs 0-3, decode 4-7).
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item4.sh Cptpcfix-1node 100 score arm-C-prod-ptpcfix.env 104
