#!/usr/bin/env bash
# main-u26: u26-rocm-exact-pin-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u26-rocm-exact-pin-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
BRANCH=u26-rocm-exact-pin-main
TESTS=(tests/v1/kv_offload/cpu)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
