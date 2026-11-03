#!/usr/bin/env bash
# n06-33 GPU 0-3, after ref-mxfp4-stock: FP8 KV only (isolates the KV dtype), then a repeat of ref-mxfp4-stock (run-to-run noise).
while docker inspect vfixref-mxfp4-stock >/dev/null 2>&1; do sleep 20; done
LONGCTX_SPECS= bash /perf_apps/liyingli/bench_agentx/vfix-aus-20261103/scripts/ref_run3.sh mxfp4-fp8kv 0,1,2,3 /perf_apps/data/models/GLM-5.2-MXFP4 -- --kv-cache-dtype fp8_e4m3
LONGCTX_SPECS= bash /perf_apps/liyingli/bench_agentx/vfix-aus-20261103/scripts/ref_run3.sh mxfp4-stock-r2 0,1,2,3 /perf_apps/data/models/GLM-5.2-MXFP4
