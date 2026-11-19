#!/usr/bin/env bash
# 37 again with the ZMQ socket-limit fix (37 lost every KV pull to "Too many open files": a TP4 decode rank pulls for
# every request, 815 at once, over libzmq's 1023 sockets per context). Decode TP4, golden 3.51, formal c768, after a
# correctness gate with block acceptance (also checks the prefill TP1 -> decode TP4 KV transfer, unverified so far).
export NP=2 DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_mc_decode_tp4_v2.sh
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="GOLDEN_AL=3.51"
export DECODE_DP_ATTN=false SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NOTE="decode TP4 golden 3.51 (37 + ZMQ MAX_SOCKETS fix); prefill DP2 x 2 unchanged"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
