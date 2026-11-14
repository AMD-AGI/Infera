#!/bin/bash
# DRIVER619-20261113 formal delivery: 65 with the roles swapped (opt_item5.sh run on the 6.16 node: prefill there, decode on
# the 6.19 node). Reference: 31 (preloaded shim): 6/6, 197/200, garbled 0, acceptance 3.51.
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 --exclude=$NODE619 env IMAGE=infera-sglang:v0522-fdbafb60-hipgem \
    ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh native-pp8-gate-b 50 gate arm-C-1p1d-pp8-gemkv-native.env
