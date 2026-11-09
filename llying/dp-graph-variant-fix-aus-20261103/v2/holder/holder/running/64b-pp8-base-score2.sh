#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 (retry: 63b and 64 failed at decode start-up on n03-33, scheduler_6 / scheduler_4 SIGSEGV after graph capture): the plain PP8 arm (arm-C-1p1d-pp8, 56,593 on n04-25 -> n04-29 on 11-06) on the same nodes as
# 63b, so the kopt / no-cap scores compare without node variance (62 ran its prefill on n04-33: full chunk 339 vs 324 ms).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh pp8-base-score2 125 score arm-C-1p1d-pp8.env 208
