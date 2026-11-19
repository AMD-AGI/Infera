#!/usr/bin/env bash
# Fallback for the decode DP4 stall (32c / 33d): the same 2P1D point with the decode as one TP4 engine (no DP ranks,
# no dummy batches), golden acceptance 3.51, formal AgentX c768. Also the TP4-vs-DP4 decode comparison point.
export NP=2 DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_mc_decode_tp4_golden.sh
export DECODE_DP_ATTN=false SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NOTE="decode TP4 (one engine, no DP), golden 3.51; prefill DP2 x 2 unchanged"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v1.sh
