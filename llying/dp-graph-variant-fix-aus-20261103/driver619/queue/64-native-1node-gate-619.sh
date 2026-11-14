#!/bin/bash
# DRIVER619-20261113 formal delivery: one-node 4+4 production C arm with native GEM-backed KV (no preload,
# arm-C-prod-ptpcfix-nohip-gemkv-native.env) on the 6.19 node (srun -w plus --exclude: a bare -w landed on the other
# node twice on 11-13).
# Reference: 46 (same arm with the preloaded shim, 6.19): 6/6, 196/200, garbled 0, acceptance 3.55.
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env SLURM_JOB_NODELIST=$NODE619 IMAGE=infera-sglang:v0522-fdbafb60-hipgem ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 \
    LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh native-1node-gate-619 45 gate arm-C-prod-ptpcfix-nohip-gemkv-native.env
