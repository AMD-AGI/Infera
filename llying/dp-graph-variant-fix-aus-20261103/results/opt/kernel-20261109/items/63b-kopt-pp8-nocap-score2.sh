#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 (rerun: 63 lost its AgentX to the 14:13 preemption of job 35372): PP8 C208 score with #1 + #2 only (no KV cap):
# separates the KV-cap effect (decode prealloc-wait 4.0 -> 11.9) from node variance. The kopt gate (60) covers these
# kernels; the decode KV path without the cap is production's.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-nocap-score2 125 score arm-C-1p1d-pp8-kopt-nocap.env 208
