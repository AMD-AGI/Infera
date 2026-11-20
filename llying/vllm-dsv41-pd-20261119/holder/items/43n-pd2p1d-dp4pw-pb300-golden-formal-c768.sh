#!/usr/bin/env bash
# Decode DP4 golden with PIECEWISE-only CUDA graphs (breakable on): attention runs as an eager break, where ROCm does not
# fork side streams, and the mHC gate forks only inside FULL capture, so no in-graph multi-stream branch remains while
# the MoE / dense layers stay in graphs. Eager DP4 (43x) ran a full formal point; every FULL-graph DP4 golden run
# stalled (README 4.4). Prefill with B300's settings. Correctness gate first, then golden 3.51, formal c768.
C=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs
export NP=2 PCFG=$C/pd1n_dpa_prefill_dp2_v2.sh DCFG=$C/pd1n_dpa_decode_dp4_v2.sh
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
PW='D_EXTRA_ARGS=--compilation-config={"cudagraph_mode":"PIECEWISE"}'
export PROBE_FIRST=1 D_PROBE_ENV="$PW" D_ENV="SPEC_GOLDEN=1 $PW"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode DP4 golden 3.51, PIECEWISE-only CUDA graphs; prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
