#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP8 decode traces of arm-C-1p1d-pp8-megatp-mt4 (MegaMoeTP lb_mt 4, lb_npp 2) with
# the synthetic load of 69a5s / 69a5u (concurrency 96 / 160 / 192): per-layer MegaMoeTP time in the decode graphs against
# 181 / 222 / 236 us with the defaults and production MoE + comm 176 / 235 / 262 us.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-mt4-pp8-strace 35 strace arm-C-1p1d-pp8-megatp-mt4-trace.env
