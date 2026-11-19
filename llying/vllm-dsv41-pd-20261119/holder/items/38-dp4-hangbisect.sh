#!/usr/bin/env bash
# Bisect the decode DP4 warmup stall (README 4.4) with a synthetic warmup-shaped load (scripts/pd_hangbisect_v1.sh):
# V0 unchanged (must stall), V1 no breakable CUDA graph, V2 block (real) acceptance, V3 eager decode.
export VARIANTS="V0 V1nobreak V2block V3eager" ROUNDS=2 LOAD_IN=24000 LOAD_N=2500 LOAD_C=640
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_hangbisect_v1.sh
