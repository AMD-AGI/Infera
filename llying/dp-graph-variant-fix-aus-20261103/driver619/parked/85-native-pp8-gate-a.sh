#!/bin/bash
# DRIVER619-20261113 formal delivery (65 again, after 80): 1P1D PP8 production arm with native GEM-backed KV (arm-C-1p1d-pp8-gemkv-native.env),
# prefill on the 6.19 node (opt_item5.sh run there), decode on the 6.16 node. Mooncake's HIP transport stays on and
# MOONCAKE_DISABLE_HIP_DMABUF stays 1, as in production.
# Reference: 30 (preloaded shim): 6/6, 194/200, garbled 0, acceptance 3.69; arm-C-1p1d-pp8 on 6.16: 197/200, 3.60.
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env IMAGE=infera-sglang:v0522-fdbafb60-hipgem \
    ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh native-pp8-gate-a 50 gate arm-C-1p1d-pp8-gemkv-native.env
