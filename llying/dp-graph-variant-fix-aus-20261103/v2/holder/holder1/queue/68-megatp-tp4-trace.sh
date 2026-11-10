#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: one-node 4+4 C104 decode trace with MegaMoeTP (AgentX 1500 s, two 40-step
# profiles): the in-graph MegaMoeTP time per layer and call, and whether other kernels overlap it. Compare with
# 57 (kdeep_tp4_kopt_tp4: KF MoE gemm1 144.9 + gemm2 75.3 + all-gather 31.5 + reduce-scatter 27.6 per layer).
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh megatp-tp4-trace 65 trace arm-C-prod-ptpcfix-nohip-megatp-trace.env 104
