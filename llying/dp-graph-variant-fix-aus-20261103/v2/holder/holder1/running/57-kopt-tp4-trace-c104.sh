#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: the cumulative TP4 kernel changes in the decode graph: one-node 4+4 C104 decode trace
# (AgentX 1500 s, two 40-step profiles). Compare per-call kernel times with opt-traceC-c104 / opt-kopt-absorb-trace-c104
# at equal graph batch sizes (ana/kbybs.py).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-tp4-trace 65 trace arm-C-prod-ptpcfix-nohip-kopt-tp4-trace.env 104
