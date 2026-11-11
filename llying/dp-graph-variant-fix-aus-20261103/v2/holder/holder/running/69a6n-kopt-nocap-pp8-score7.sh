#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: same-allocation A/B for the tuned MegaMoeTP, part 1: kopt-nocap C208 3600 s on the
# same node pair right before 69a6o. 69a6m (salvaged) vs 64c: decode ITL -3 / -7 %, interactivity +2.7 / +7.3 %, but
# throughput -3.9 % with the same prefill node 11 % slower per chunk (323 -> 358 ms), so throughput needs the same nodes.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-pp8-nocap-score7 125 score arm-C-1p1d-pp8-kopt-nocap.env 208
