#!/usr/bin/env bash
# Decode DP4 golden with CUDA graphs and only the mHC gate fork turned off (vllm-overlay/streams/mhc.py,
# VLLM_ROCM_DISABLE_MULTI_STREAM_OVERLAP=1; platforms/rocm.py is not overlaid, so the attention forks inside capture
# stay: turning those off too broke the capture in 43q). Tests half of the in-graph multi-stream hypothesis (README 4.4).
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export MOUNTS="-v /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/vllm-overlay/streams/mhc.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mhc.py:ro"
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_WENV="VLLM_ROCM_DISABLE_MULTI_STREAM_OVERLAP=1"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode DP4 golden 3.51, mHC in-graph overlap off (overlay); attention overlap unchanged; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
