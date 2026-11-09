#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP4 C104 score of the production C arm with only MegaMoeTP (gate 64: 197/200,
# 16k 64/64, acceptance 3.57), the nohip arm's 1500 s window; 66 scores the production C arm right after on the same node.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh score1500-megatp 60 score arm-C-prod-ptpcfix-nohip-megatp.env 104
