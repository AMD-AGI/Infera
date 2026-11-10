#!/usr/bin/env bash
# REPORT-20261110 section 6: amdgpu 6.19 start-up segfault bisect on the pinned node: one-node 4+4 gate with the Mooncake knob of arm-C-prod-ptpcfix-nohip-mc-serialreg.env, one start-up attempt, short probe.
ALLOW_PILOT_DRIVER=1 UP_ATTEMPTS=1 PROBE_N=50 LONGCTX_SPECS="16000:16" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh mc-serialreg-gate 25 gate arm-C-prod-ptpcfix-nohip-mc-serialreg.env
