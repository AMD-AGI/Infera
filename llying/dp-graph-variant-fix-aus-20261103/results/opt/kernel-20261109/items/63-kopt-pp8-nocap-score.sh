#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: 1P(PP8)+1D C208 score with #1 + #2 only (no KV cap), same nodes as 62 (kopt, 53,341):
# separates the KV-cap effect (decode prealloc-wait 4.0 -> 11.9) from node variance. The kopt gate (60) covers these
# kernels; the decode KV path without the cap is production's.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-nocap-score 125 score arm-C-1p1d-pp8-kopt-nocap.env 208
