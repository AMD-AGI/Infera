#!/usr/bin/env bash
# REPORT-20261110 section 6: does peer-memory KV registration (MOONCAKE_DISABLE_HIP_DMABUF=1, kit README note 5) fix the
# start-up segfault on an amdgpu 6.19.14 firmware-pilot node? One-node 4+4 production C arm with only that change, on the
# pinned debug job (n03-33: three decode segfaults with the default dma-buf registration on 11-09).
# Reference (production C arm on 6.16.13 nodes): smoke 6/6, 196-197/200, garbled 0, acceptance 3.60-3.62.
ALLOW_PILOT_DRIVER=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh dmabuf1-gate-pilot 45 gate arm-C-prod-ptpcfix-nohip-dmabuf1.env
