#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: one-node 4+4 (TP4 decode) synthetic-load decode traces of the MegaMoeTP tuned at TP4 (lb_mt 2, npp 2, q 2, lb_min 64), concurrency
# 48 / 80 / 96 (per-rank batch ~12 / 20 / 24 over 4 DP ranks), per-layer MoE + comm with scripts/ana/moe_bybs.py --layers 75.
# Pair: 61c production / 61d tuned MegaMoeTP (lb_min 256) / 61f with lb_min 64. Needs a 6.16 node (PD start-up fails on the 6.19 pilot driver).
STAGES=48:150,80:150,96:150 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-tp4-mt2q2-lbmin64-strace 35 strace arm-C-prod-ptpcfix-nohip-megatp-mt2q2-trace.env
