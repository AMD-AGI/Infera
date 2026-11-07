#!/usr/bin/env bash
# v2 kit + v2 image on two nodes, 4+4 (prefill GPUs 4-7 here, decode 4-7 on the other node) at C104: against 38b
# (same pair, fdbafb60 package) for the image, and against one node (41,124 / 41,395) for colocation.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-4p4-c104 sweep 4p4 104
