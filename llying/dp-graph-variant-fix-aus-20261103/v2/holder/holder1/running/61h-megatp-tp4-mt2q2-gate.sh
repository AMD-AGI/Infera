#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: end-to-end PD gate of the TP4 tuned MegaMoeTP arm (arm-C-prod-ptpcfix-nohip-megatp-mt2q2:
# lb_mt 2, npp 2, q 2, lb_min 64; op SNR unchanged). Reference: 64 (default MegaMoeTP TP4) 197/200, 16k 64/64, acceptance 3.57;
# production C arm 196-197/200, acceptance 3.60-3.62.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-tp4-mt2q2-gate 45 gate arm-C-prod-ptpcfix-nohip-megatp-mt2q2.env
