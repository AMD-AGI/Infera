#!/usr/bin/env bash
# C + PTPC/#45800 at 4+4 C104 again on this holder's node pair (53,666 was n06-25 -> n06-33): the same-pair reference
# for the dense FP8 score (39) and the v2 kit (40).
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item3.sh Cptpcfix-recal 100 score arm-C-prod-ptpcfix.env 104
