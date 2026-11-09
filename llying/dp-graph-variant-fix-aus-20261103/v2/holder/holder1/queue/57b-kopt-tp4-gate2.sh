#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 (rerun: 56 was cut short by the 10:33 preemption of job 35370): correctness of the cumulative TP4 kernel changes (absorb configs + fused2 K3 grouping +
# KF MoE gate 416) on one-node 4+4: smoke, 200 GSM8K probe, 16k x 64, real MTP acceptance.
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh kopt-tp4-gate2 45 gate arm-C-prod-ptpcfix-nohip-kopt-tp4.env
