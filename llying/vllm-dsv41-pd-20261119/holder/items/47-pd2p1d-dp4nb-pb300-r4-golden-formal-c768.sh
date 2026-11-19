#!/usr/bin/env bash
# 42 + --renderer-num-workers 4 on both legs (DSv4.1 encodes prompts in Python; GLM-5.2 on the same stack spent 1.35 s
# per request rendering on one thread, vllm_glm52_agentx D36).
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64 P_EXTRA_ARGS=--renderer-num-workers=4"
export D_WENV="VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export PROBE_FIRST=1 D_PROBE_ENV="D_EXTRA_ARGS=--renderer-num-workers=4" D_ENV="SPEC_GOLDEN=1 D_EXTRA_ARGS=--renderer-num-workers=4"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NEED_MIN=130
export NOTE="42 + renderer 4 threads on both legs"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
