#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: correctness gate of the tuned MegaMoeTP arm (arm-C-1p1d-pp8-megatp-mt4: lb_mt 4,
# lb_npp 2, lb_q 2; tiling only, op SNR against production unchanged at 29.1 dB) before production adoption: smoke,
# 200-prompt GSM8K probe, 16k long-context probe, real acceptance. Reference: 65a (default MegaMoeTP) and arm-C-1p1d-pp8
# (smoke 6/6, 197/200, bad_start 0, garbled 0, acceptance 3.60).
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-mt4q2-pp8-gate 50 gate arm-C-1p1d-pp8-megatp-mt4.env
