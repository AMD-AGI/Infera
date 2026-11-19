#!/usr/bin/env bash
# Diagnostic for the decode DP4 warmup stall: the unchanged DP4 decode (breakable CUDA graphs on) with block (real)
# acceptance instead of golden, AgentX formal c768 (not a scored configuration). Every stall so far was in the step
# that is a max_tokens=1 request's first and last decode step, whose drafts are stale slot data that golden acceptance
# accepts at its fixed rates; if this warmup passes, golden acceptance is the trigger.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_ENV=""
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN="" MODE=agentx FAST=0 NEED_MIN=60
export NOTE="DIAG: decode DP4 default graphs, block (real) acceptance; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
