#!/usr/bin/env bash
# v2 kit 1P2D (one PP4 prefill, two TP4 / DP4 decodes on the other node; 12 GPUs): smoke (router lists 2 decodes),
# GSM8K probe 200, 16k x 64, real acceptance. First run of DECODE_WORKERS=2 (31-v2-1p2d-sweep runs only if this is clean).
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-1p2d-gate gate 1p2d
