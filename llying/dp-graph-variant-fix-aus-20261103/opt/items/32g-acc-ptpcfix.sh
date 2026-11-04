#!/usr/bin/env bash
# PTPC with #45800: real acceptance and correctness (smoke + 200-prompt probe + 16k x 64).
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh acc-ptpcfix 30 gate arm-C-prod-ptpcfix.env
