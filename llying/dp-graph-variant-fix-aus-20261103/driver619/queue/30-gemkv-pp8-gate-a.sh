#!/bin/bash
# DRIVER619-20261113 section 6 step 3: GLM-5.2 1P1D PP8 production arm with GEM-backed KV pools (arm-C-1p1d-pp8-gemkv),
# prefill on the batch host, decode on the other node of the 6.19 + 6.16 pair.
# Reference (arm-C-1p1d-pp8, 11-05, 6.16 nodes): smoke 6/6, 197/200, bad_start 0, garbled 0, acceptance 3.60.
echo "batch host $(hostname -s) amdgpu $(cat /sys/module/amdgpu/version); NODE619=$NODE619 NODE616=$NODE616"
ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh gemkv-pp8-gate-a 50 gate arm-C-1p1d-pp8-gemkv.env
