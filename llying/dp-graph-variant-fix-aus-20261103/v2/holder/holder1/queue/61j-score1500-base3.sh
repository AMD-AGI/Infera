#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: same-node reference for 61i (production C arm, TP4 C104, 1500 s window).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh score1500-base3 60 score arm-C-prod-ptpcfix-nohip.env 104
