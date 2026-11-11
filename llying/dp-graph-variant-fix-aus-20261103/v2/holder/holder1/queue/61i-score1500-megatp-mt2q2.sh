#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP4 C104 score (1500 s window) of the tuned MegaMoeTP arm; 61j scores the production
# C arm right after on the same node. In-graph (61c / 61f): verify step +0.7 / -3.0 / -3.8 % at per-rank batch 12 / 20 / 24.
# Earlier same-node pair with the default MegaMoeTP: 49,016 (65b) vs 49,242 (66).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh score1500-megatp-mt2q2 60 score arm-C-prod-ptpcfix-nohip-megatp-mt2q2.env 104
