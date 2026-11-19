#!/usr/bin/env bash
# Decode DP4 with expert parallelism (--enable-expert-parallel: experts spread over the 4 ranks instead of every expert
# tensor-split 4 ways; B300's decode is DEP8) and golden 3.51, formal c768. The MoE-TP DP4 decode stalls under golden
# acceptance in every configuration tried (README 4.4); EP changes the MoE communication pattern. Gate first.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export PROBE_FIRST=1 D_PROBE_ENV="D_EXTRA_ARGS=--enable-expert-parallel" D_ENV="SPEC_GOLDEN=1 D_EXTRA_ARGS=--enable-expert-parallel"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode DP4 + EP4, golden 3.51; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
