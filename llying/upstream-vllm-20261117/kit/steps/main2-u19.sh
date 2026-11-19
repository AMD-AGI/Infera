#!/usr/bin/env bash
# main2-u19: u19-rocm-pin-worker-slots-main rebased onto main 8828834fb vs that main (wt/main2), nightly image with the
# trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u19-rocm-pin-worker-slots-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main2.sh
BRANCH=u19-rocm-pin-worker-slots-main
TESTS=(tests/v1/kv_offload/cpu/test_shared_offload_region.py tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main2.sh
