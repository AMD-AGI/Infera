#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 #2: correctness of the MLA absorb BMM decode configs on one-node 4+4 (smoke, 200 GSM8K probe,
# 16k x 64 long context, real MTP acceptance) before any score with them.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-absorb-gate 45 gate arm-C-prod-ptpcfix-nohip-kopt-absorb.env
