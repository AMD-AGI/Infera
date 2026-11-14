#!/bin/bash
# DRIVER619-20261113 section 6 step 3: the one-node 4+4 production C arm (Mooncake HIP route off, same-NIC RDMA loopback)
# with GEM-backed KV, entirely on the 6.19 node (the e1ab 4+4 deployment that segfaulted on n04-25 on 11-12).
# opt_item5.sh picks one-node mode from the node list, so it gets only the 6.19 node.
# Reference (arm-C-prod-ptpcfix-nohip on 6.16.13): smoke 6/6, 196-197/200, garbled 0, acceptance 3.60-3.62.
printf '. "$PKG/config/arm-C-prod-ptpcfix-nohip.env"\n. "$PKG/config/gemkv.env"\n' \
    > /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/config/arm-C-prod-ptpcfix-nohip-gemkv.env
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 env SLURM_JOB_NODELIST=$NODE619 ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 \
    PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh gemkv-1node-gate 45 gate arm-C-prod-ptpcfix-nohip-gemkv.env
