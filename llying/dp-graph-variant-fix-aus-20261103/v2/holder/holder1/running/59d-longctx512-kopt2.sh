#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: the 16k long-prefix GSM8K runaway rate (kopt gates: 2/32 twice; baselines 0/128), larger sample:
# 16000 x 512, no probe; repeat in reverse order (kopt first, then base) to see the MTP acceptance spread (3.46 base vs 3.36 kopt in 59a / 59b); kopt (arm-C-prod-ptpcfix-nohip-kopt-tp4.env).
PROBE_N=0 LONGCTX_SPECS="16000:512" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh longctx512-kopt2 35 gate arm-C-prod-ptpcfix-nohip-kopt-tp4.env
