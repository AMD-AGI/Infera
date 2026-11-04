#!/usr/bin/env bash
# 2P1D C208 with the prefill policy hrrn (vs shortest-prefill-first: 54,094, > 128K prompts TTFT p99 1,017 s).
exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh 2p1d-hrrn 115 score arm-C-2p1d-ptpcfix-hrrn.env 208
