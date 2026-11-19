#!/usr/bin/env bash
# Diagnostic for the decode DP4 warmup stall: DP4 decode with --enforce-eager (no CUDA graphs at all), golden 3.51,
# AgentX formal c768. If this warmup passes where 32c / 33d / 36 / 42 stalled, graph replay is involved.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_ENV="SPEC_GOLDEN=1 D_EXTRA_ARGS=--enforce-eager"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=60
export NOTE="DIAG: decode DP4 eager, golden 3.51; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
