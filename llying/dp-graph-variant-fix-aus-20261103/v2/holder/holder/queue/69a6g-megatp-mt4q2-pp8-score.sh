#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP8 MegaMoeTP (lb_mt 4, lb_npp 2, lb_q 2) C208 score, full 3600 s window, against 64c
# kopt-nocap 56,595 (decode step by busiest-rank running and interactivity). Replaces 69a6e (default MegaMoeTP config).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-mt4q2-pp8-score 125 score arm-C-1p1d-pp8-megatp-mt4.env 208
