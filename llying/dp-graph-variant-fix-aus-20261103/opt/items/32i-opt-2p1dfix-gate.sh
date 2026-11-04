#!/usr/bin/env bash
# 2P1D + PTPC/#45800: draft acceptance at TP8 and correctness before the C208 score.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh 2p1dfix-gate 40 gate arm-C-2p1d-ptpcfix.env
