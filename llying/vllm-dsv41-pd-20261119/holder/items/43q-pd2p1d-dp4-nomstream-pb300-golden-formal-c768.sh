#!/usr/bin/env bash
# Decode DP4 golden with CUDA graphs but without the in-graph multi-stream overlap (VLLM_ROCM_DISABLE_MULTI_STREAM_OVERLAP=1
# via vllm-overlay/streams/: the attention compressor / indexer forks and the mHC gate fork, which ROCm only enables inside
# graph capture because runtime HIP event waits can hang). The eager DP4 decode (43x) got past the point where every
# graph-mode golden DP4 run stalled (32c / 33d / 36 / 42 / 44 / 43r / 43w). Prefill with B300's settings. Gate first.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export MOUNTS="-v /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/vllm-overlay/streams/rocm.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro -v /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/vllm-overlay/streams/mhc.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mhc.py:ro"
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_WENV="VLLM_ROCM_DISABLE_MULTI_STREAM_OVERLAP=1"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode DP4 golden 3.51, CUDA graphs without in-graph multi-stream overlap (overlay); prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
