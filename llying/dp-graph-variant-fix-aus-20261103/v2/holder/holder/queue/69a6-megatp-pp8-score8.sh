#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP8 MegaMoeTP C208 score with the full 3600 s window, compared with the existing 3600 s
# references (64c kopt-nocap 56,595 on n01-29 -> n02-21; PP8 production 56,593 on n04-25 -> n04-29): mainly the decode step by
# busiest-rank running and interactivity, which vary little across nodes. A same-node reference only if the result is ambiguous.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-pp8-score8 125 score arm-C-1p1d-pp8-megatp.env 208
