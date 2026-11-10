#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: the tuned MegaMoeTP score (69a6g) stalled at the AgentX warm-up tail (9 requests
# left, 0-1 per DP rank; nothing completed for 15 min; first such stall in any opt run). Low-concurrency synthetic load on
# the tuned arm: concurrency 8 / 24 / 40 / 64 (per-rank batch ~1 / 3 / 5 / 8, 32-256 global tokens, below the 384 that
# every earlier test used). A stall shows as a flat done= count in trace/opt-megatp-mt4q2-lowbs-strace/strace.log.
STAGES=8:150,24:150,40:150,64:150 ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-mt4q2-lowbs-strace 40 strace arm-C-1p1d-pp8-megatp-mt4-trace.env
