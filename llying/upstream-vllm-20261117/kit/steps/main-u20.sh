#!/usr/bin/env bash
# main-u20: u20-simple-offload-page-align-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u20-simple-offload-page-align-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
BRANCH=u20-simple-offload-page-align-main
TESTS=(tests/utils_/test_host_memory.py tests/v1/simple_kv_offload tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py tests/v1/worker/test_utils.py)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
