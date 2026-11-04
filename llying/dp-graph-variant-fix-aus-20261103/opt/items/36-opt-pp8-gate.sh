#!/usr/bin/env bash
# 1P(PP8)+1D(TP8): first deployment of a PP8 prefill on this stack; correctness and acceptance before scoring.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh pp8-gate 50 gate arm-C-1p1d-pp8.env
