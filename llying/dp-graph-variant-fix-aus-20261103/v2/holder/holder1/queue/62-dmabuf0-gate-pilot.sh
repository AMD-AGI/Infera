#!/usr/bin/env bash
# REPORT-20261110 section 6: control for 61 on the same amdgpu 6.19.14 node: the unchanged one-node production C arm
# (default dma-buf KV registration). Expected to segfault at start-up if dma-buf registration is the cause.
ALLOW_PILOT_DRIVER=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh dmabuf0-gate-pilot 45 gate arm-C-prod-ptpcfix-nohip.env
