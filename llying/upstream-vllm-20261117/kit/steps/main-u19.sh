#!/usr/bin/env bash
# main-u19: u19-rocm-pin-worker-slots-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u19-rocm-pin-worker-slots-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
BRANCH=u19-rocm-pin-worker-slots-main
TESTS=(tests/v1/kv_offload/cpu/test_shared_offload_region.py tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
