#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: same-allocation reference for 64h: arm-C-1p1d-pp8-kopt-nocap at C208
# (64c on n01-29 -> n02-21: 56,595; PP8 baseline 11-06: 56,593).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-nocap-score4 125 score arm-C-1p1d-pp8-kopt-nocap.env 208
