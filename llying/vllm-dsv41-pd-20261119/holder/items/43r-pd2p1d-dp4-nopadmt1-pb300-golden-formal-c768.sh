#!/usr/bin/env bash
# Decode DP4 golden with the P/D first-step padding dropped for requests that can emit only one more token
# (VLLM_PD_PAD_FIRST_STEP_MIN_REMAINING=2, scheduler overlay vllm-overlay/sched/scheduler.py). Every DP4 golden stall
# (32c / 33d / 36 / 42 / 44) was in such a step: a max_tokens=1 warmup request's first decode step, padded to 1 + 5
# rows to keep the full CUDA graph; with block acceptance (43s) the whole warmup passed. Breakable CUDA graphs on (the
# default; turning them off did not help, 42). Prefill with B300's settings. Gate first, then golden 3.51, c768.
export NP=2 PCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_prefill_dp2_v2.sh DCFG=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/configs/pd1n_dpa_decode_dp4_v2.sh
export MOUNTS="-v /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/vllm-overlay/sched/scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro"
export P_ENV="P_UTIL=0.93 P_LONG_PREFILL=8192 P_MAXSEQS=64"
export D_WENV="VLLM_PD_PAD_FIRST_STEP_MIN_REMAINING=2"
export PROBE_FIRST=1 D_PROBE_ENV="" D_ENV="SPEC_GOLDEN=1"
export DECODE_DP_ATTN=true SIM_ACCEPT_LEN=3.51 MODE=agentx FAST=0 NEED_MIN=130
export NOTE="decode DP4 golden 3.51, no first-step padding for max_tokens-1 requests (scheduler overlay); prefill B300 settings"
exec bash /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/pd_run_v2.sh
