#!/usr/bin/env bash
# A point that can finish while decode DP4 stalls (42 stalled at warmup min 6 with breakable CUDA graphs off, so that
# was not the fix): decode one TP4 engine with more KV room than 39 / 40 (36.1M tokens, 99.8% used in 40's warmup):
# Engram tables in host memory (agg_tp4 keeps them in HBM), --gpu-memory-utilization 0.93, --max-num-seqs 256.
# Prefill with B300's settings (util 0.93, long-prefill 8192, max-num-seqs 64). Correctness gate (block acceptance)
# first, then golden 3.51, formal AgentX c768.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_mc_decode_tp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export PROBE_FIRST=1 D_PROBE_ENV="D_ENGRAM_HOST=1 D_UTIL=0.93 D_MAXSEQS=256" D_ENV="D_ENGRAM_HOST=1 D_UTIL=0.93 D_MAXSEQS=256 GOLDEN_AL=3.51"
export DECODE_DP_ATTN=false SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode TP4 + Engram on host + util 0.93 + max-num-seqs 256, golden 3.51; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
