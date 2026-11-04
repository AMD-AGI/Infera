#!/usr/bin/env bash
# Dense-layer online FP8 production correctness on the one-node holder, after the _is_block_scale_fp8 fix (05 failed: assert in apply_fp8_linear) (pkg kit, prefill GPUs 0-3 + decode 4-7 on one
# node; correctness does not depend on the shape), so the two-node holder can go to the score (39) directly.
PROBE_N=1319 LONGCTX_SPECS="16000:128 100000:32" EVAL=1 exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item4.sh densefp8-prod 95 gate arm-C-prod-ptpcfix-densefp8.env
