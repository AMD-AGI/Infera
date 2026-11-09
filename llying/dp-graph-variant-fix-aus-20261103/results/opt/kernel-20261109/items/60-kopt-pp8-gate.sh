#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 #1-#3 on 1P(PP8)+1D(TP8): correctness and real acceptance before the trace / score.
# Reference (arm-C-1p1d-pp8, 11-05): smoke 6/6, 197/200, bad_start 0, garbled 0, acceptance 3.60.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-gate 50 gate arm-C-1p1d-pp8-kopt.env
