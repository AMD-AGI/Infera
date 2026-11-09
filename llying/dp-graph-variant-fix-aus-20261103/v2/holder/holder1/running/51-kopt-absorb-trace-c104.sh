#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 #2 in the decode graph: one-node 4+4 C104 decode trace (AgentX 1500 s, two 40-step profiles)
# with the two MLA absorb BMM decode configs mounted. Compare per-call absorb kernel times with opt-traceC-c104 at equal
# graph batch sizes (kdeep.py).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-absorb-trace 65 trace arm-C-prod-ptpcfix-nohip-kopt-absorb-trace.env 104
