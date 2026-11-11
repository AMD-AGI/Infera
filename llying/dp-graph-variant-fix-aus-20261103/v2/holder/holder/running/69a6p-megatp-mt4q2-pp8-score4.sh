#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: second same-node A/B on the same pair, order reversed (MegaMoeTP first, then 69a6q
# kopt-nocap): the first pair (69a6n / 69a6o) differed by +0.4 % in throughput; a second sample gives the same-node noise
# and rules out an order effect.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-mt4q2-pp8-score4 125 score arm-C-1p1d-pp8-megatp-mt4.env 208
