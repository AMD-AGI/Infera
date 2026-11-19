#!/usr/bin/env bash
# main-u20: u20-simple-offload-page-align-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u20-simple-offload-page-align-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
BRANCH=u20-simple-offload-page-align-main
# Pre-existing on main (nightly, MI355X): test teardowns in these files abort at random in torch's empty_host_cache
# (base too), so each file runs in its own process (PER_FILE) and a file whose process died is retried once.
PER_FILE=1
TESTS=(tests/utils_/test_host_memory.py tests/v1/simple_kv_offload/test_worker_layout.py tests/v1/simple_kv_offload/test_host_buffer.py tests/v1/simple_kv_offload/test_scheduler.py tests/v1/simple_kv_offload/test_worker.py tests/v1/simple_kv_offload/test_integration.py tests/v1/simple_kv_offload/test_kv_events.py tests/v1/simple_kv_offload/test_hip_mem_ops.py tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py tests/v1/worker/test_utils.py)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
