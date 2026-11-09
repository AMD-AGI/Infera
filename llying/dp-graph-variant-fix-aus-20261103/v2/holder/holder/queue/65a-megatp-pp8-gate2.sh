#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP8 decode MoE layers as aiter's MegaMoeTP (arm-C-1p1d-pp8-megatp, on top of
# 64c's #1 + #2): correctness and real acceptance before the trace / score. The decode log's "[kopt-megatp]" lines show
# whether the path engaged (instance, shapes, topk columns).
# Reference (arm-C-1p1d-pp8, 11-05): smoke 6/6, 197/200, bad_start 0, garbled 0, acceptance 3.60; 60 (kopt): 197/200, 3.59.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-pp8-gate2 50 gate arm-C-1p1d-pp8-megatp.env
