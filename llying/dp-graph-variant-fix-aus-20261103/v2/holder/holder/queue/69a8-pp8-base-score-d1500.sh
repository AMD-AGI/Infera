#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: production PP8 reference with the same 1500 s window and allocation as 69a6 (MegaMoeTP) / 69a7 (kopt-nocap).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh pp8-base-score-d1500 70 score arm-C-1p1d-pp8-d1500.env 208
