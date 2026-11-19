#!/usr/bin/env bash
# 42's configuration at concurrency 1024 (the higher-throughput end of the pareto curve).
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_WENV="VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NEED_MIN=130
export NOTE="42 at c1024"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
