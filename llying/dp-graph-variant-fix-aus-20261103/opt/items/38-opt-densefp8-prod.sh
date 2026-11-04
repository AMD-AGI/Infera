#!/usr/bin/env bash
# Dense-layer online FP8 (arm-C-prod-ptpcfix-densefp8): the full production correctness suite (1,319 prompts, 16k x 128,
# 100k x 32, lm-eval) against C + PTPC/#45800 (0.971, 64/64 62/64, 16/16 16/16, lm-eval 0.9697 / 0.9689, accept 3.62).
PROBE_N=1319 LONGCTX_SPECS="16000:128 100000:32" EVAL=1 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item3.sh densefp8-prod 95 gate arm-C-prod-ptpcfix-densefp8.env
