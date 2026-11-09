#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: attribute the long-context MTP acceptance gap (kopt-tp4 3.36 / 3.36 vs production 3.46 / 3.41,
# 16000 x 512): absorb (arm-C-prod-ptpcfix-nohip-kopt-absorb.env).
PROBE_N=0 LONGCTX_SPECS="16000:512" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh longctx512-absorb 35 gate arm-C-prod-ptpcfix-nohip-kopt-absorb.env
