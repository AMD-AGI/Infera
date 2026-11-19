#!/usr/bin/env bash
# main-u11: u11-offload-pp-worker-bytes-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u11-offload-pp-worker-bytes-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
BRANCH=u11-offload-pp-worker-bytes-main
TESTS=(tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_kv_cache_planning.py tests/v1/kv_connector/unit/offloading_connector tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
