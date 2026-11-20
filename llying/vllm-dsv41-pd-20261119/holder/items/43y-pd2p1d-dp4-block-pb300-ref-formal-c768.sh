#!/usr/bin/env bash
# Reference (not scored): the full 43s run, decode DP4 with block (real) acceptance, which passes the AgentX warmup
# where every golden DP4 run stalls. AgentX's real acceptance length (3.44 in 43s) is close to golden 3.51, so this
# shows roughly where DP4 lands once the golden stall is fixed. Prefill with B300's settings, formal c768.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_ENV=""
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN="" MODE=agentx FAST=0 NEED_MIN=130
export NOTE="REFERENCE (real acceptance, not scored): decode DP4 block acceptance; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
