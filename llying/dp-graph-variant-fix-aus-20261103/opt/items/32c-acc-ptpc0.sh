#!/usr/bin/env bash
# Real-acceptance bisection (smoke + 200-prompt probe, acceptance from the decode log).
PROBE_N=200 LONGCTX_SPECS= EVAL=0 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh acc-ptpc0 25 gate arm-C-prod-ptpc0.env
