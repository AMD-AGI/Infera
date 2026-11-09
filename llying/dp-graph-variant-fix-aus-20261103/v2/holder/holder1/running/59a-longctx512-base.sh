#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: the 16k long-prefix GSM8K runaway rate (kopt gates: 2/32 twice; baselines 0/128), larger sample:
# 16000 x 512 (256 GSM8K + 256 needle), no probe; same node back to back for base (arm-C-prod-ptpcfix-nohip.env).
PROBE_N=0 LONGCTX_SPECS="16000:512" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh longctx512-base 35 gate arm-C-prod-ptpcfix-nohip.env
