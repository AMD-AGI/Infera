#!/usr/bin/env bash
# Decode DP4 warmup-stall bisect, every variant a fresh deployment (38 restarted only the decode, which the stalled
# round's prefill state then blocked). 38's V0 (unchanged) stalled in 8.5 min under this load, the same signature as
# the AgentX warmup (decode DP2's GPU stuck, DP0/1/3 in the next dummy batch). V1 no breakable CUDA graph (vLLM falls
# back to FULL_DECODE_ONLY), V2 block (real) acceptance, V3 eager decode, V5 no async scheduling.
export VARIANTS="V1nobreak V2block V3eager V5noasync" ROUNDS=2 LOAD_IN=24000 LOAD_N=2500 LOAD_C=640
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_hangbisect_v2.sh
