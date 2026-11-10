#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: same-node reference for 65 (production C arm, TP4 C104, 1500 s window;
# 59k on n02-21: 49,493 tok/s/GPU, interactivity 65.3 / 58.8).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh score1500-base2 60 score arm-C-prod-ptpcfix-nohip.env 104
