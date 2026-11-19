#!/usr/bin/env bash
# 42 with the original prefill settings (util 0.9, no long-prefill threshold, default max-num-seqs): the baseline for
# the prefill settings.
export NP=2 DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export D_WENV="VLLM_USE_BREAKABLE_CUDAGRAPH=0"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0
export NOTE="decode DP4 no breakable CUDA graph, golden 3.51; prefill unchanged"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
