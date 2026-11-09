#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: same-node A/B of the TP4 kernel package end to end (it already includes any MTP acceptance
# effect, kopt-tp4 3.38 vs production 3.43 request-weighted on 16k-only load): one-node 4+4 C104, the nohip arm's
# 1500 s window; kopt (arm-C-prod-ptpcfix-nohip-kopt-tp4.env).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh score1500-kopt 60 score arm-C-prod-ptpcfix-nohip-kopt-tp4.env 104
