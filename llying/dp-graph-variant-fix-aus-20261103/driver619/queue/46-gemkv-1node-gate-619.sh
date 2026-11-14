#!/bin/bash
# DRIVER619-20261113: 45 again, on the 6.19 node. 45 asked srun for -w <6.19 node> but Slurm ran the step on the 6.16 node
# (step 36242.46), so this runs opt_item5.sh in the holder itself (the batch host is the 6.19 node) and refuses otherwise.
# Reference (arm-C-prod-ptpcfix-nohip on 6.16.13): smoke 6/6, 196-197/200, garbled 0, acceptance 3.60-3.62;
# 45 (same arm + GEM, ran on 6.16.13): 6/6, 195/200, garbled 0, acceptance 3.64.
h=$(hostname -s)
[ "$h" = "$NODE619" ] || { echo "batch host $h is not the 6.19 node $NODE619"; exit 1; }
[[ "$(cat /sys/module/amdgpu/version)" == 6.19.* ]] || { echo "amdgpu $(cat /sys/module/amdgpu/version) on $h"; exit 1; }
SLURM_JOB_NODELIST=$NODE619 ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh gemkv-1node-gate-619 45 gate arm-C-prod-ptpcfix-nohip-gemkv.env
