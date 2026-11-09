#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: end-to-end PD gate of the MegaMoeTP overlay on one node (4+4, TP4 decode), since
# the two-node job 35398 waits for nodes. Reference (production C arm, 53 / 57b): smoke 6/6, 196-197/200, garbled 0,
# acceptance 3.60-3.62. The decode log's "[kopt-megatp]" lines show the path engaged.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-tp4-gate 45 gate arm-C-prod-ptpcfix-nohip-megatp.env
