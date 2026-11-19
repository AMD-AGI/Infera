#!/usr/bin/env bash
# 39 + B300's prefill settings: --gpu-memory-utilization 0.93 (B300 0.95), long-prefill threshold 8192, max-num-seqs 64.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_mc_decode_tp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="GOLDEN_AL=3.51"
export DECODE_DP_ATTN=false SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NOTE="39 + prefill util 0.93 / long-prefill 8192 / max-num-seqs 64 (B300 prefill knobs)"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
