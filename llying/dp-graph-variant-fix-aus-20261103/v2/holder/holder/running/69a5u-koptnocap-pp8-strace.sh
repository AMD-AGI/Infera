#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: TP8 decode traces of arm-C-1p1d-pp8-kopt-nocap at fixed per-rank batch ~12 / 20 / 24
# (synthetic load, concurrency 96 / 160 / 192), ~25 min so it fits the 30 min preemption exemption. Reference for 69a5s
# (MegaMoeTP, same load): per-layer MoE + comm time inside the decode graphs, to explain why MegaMoeTP saves only
# 1-2 ms per step in situ against ~3 ms predicted.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh koptnocap-pp8-strace 35 strace arm-C-1p1d-pp8-kopt-nocap-trace.env
