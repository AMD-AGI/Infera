#!/usr/bin/env bash
# 1P1D with the eager DP4 decode: one prefill DP2 worker (GPUs 0-1) + decode DP4 --enforce-eager (GPUs 4-7), scored on
# 6 GPUs, golden 3.51, formal c768. 43x (2P1D, the same decode) = 291,400 tok/s/GPU on 8 GPUs with both prefill engines
# about half used; the eager decode is the only DP4 mode that has not stalled under golden acceptance (README 4.4).
# Correctness gate (block acceptance, eager) first.
export NP=1 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export PROBE_FIRST=1 D_PROBE_ENV="D_EXTRA_ARGS=--enforce-eager" D_ENV="SPEC_GOLDEN=1 D_EXTRA_ARGS=--enforce-eager"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="1P1D (6 GPUs), decode DP4 eager, golden 3.51; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
