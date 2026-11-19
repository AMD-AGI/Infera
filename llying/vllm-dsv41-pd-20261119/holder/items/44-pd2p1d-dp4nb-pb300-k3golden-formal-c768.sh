#!/usr/bin/env bash
# 42 with DSpark K=3 on both legs, the decode verified with InferenceX's golden length for K=3 (2.90; the GLM-5.2
# recipe's golden 2.99 is MTP K=3). Correctness gate with block acceptance at K=3.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64 SPEC_K=3"
export D_WENV="VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export PROBE_FIRST=1 D_PROBE_ENV="SPEC_K=3" D_ENV="SPEC_K=3 GOLDEN_AL=2.90"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=2.90 MODE=agentx FAST=0
export NOTE="42 + DSpark K=3 both legs, decode golden 2.90"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
