#!/usr/bin/env bash
# Full production correctness of the best correct config (C + PTPC/#45800): 1,319 prompts, 16k x 128, 100k x 32, lm-eval.
PROBE_N=1319 LONGCTX_SPECS="16000:128 100000:32" EVAL=1 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item2.sh Cfix-prod 90 gate arm-C-prod-ptpcfix.env
