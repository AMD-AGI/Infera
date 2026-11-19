#!/usr/bin/env bash
# 1P1D: one prefill DP2 worker (GPUs 0-1) + decode DP4 (GPUs 4-7), scored on 6 GPUs (B300's prefill : decode is 4 : 8
# at c768). Concurrency 512 (85 per scored GPU; 42 is 96 at c768 on 8). 42's settings otherwise.
export NP=1 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_WENV="VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NEED_MIN=130
export NOTE="1P1D (6 GPUs) with 42's settings, c512"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
