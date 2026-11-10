#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: C208 score of arm-C-1p1d-pp8-megatp with a 1500 s window (the 125 min scores kept being preempted).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-pp8-score-d1500 70 score arm-C-1p1d-pp8-megatp-d1500.env 208
