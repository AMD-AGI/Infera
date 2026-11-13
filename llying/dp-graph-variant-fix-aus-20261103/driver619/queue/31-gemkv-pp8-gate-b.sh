#!/bin/bash
# DRIVER619-20261113 section 6 step 3: 30 with the roles swapped (opt_item5.sh run on the other node: prefill there,
# decode on the batch host), so each role has run on the 6.19 node.
b=$(hostname -s)
o=$NODE619; [ "$o" = "$b" ] && o=$NODE616
echo "batch host $b; prefill on $o"
srun --jobid=$JOB --overlap -N1 -n1 -w $o env ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh gemkv-pp8-gate-b 50 gate arm-C-1p1d-pp8-gemkv.env
