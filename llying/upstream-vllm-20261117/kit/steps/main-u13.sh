#!/usr/bin/env bash
# main-u13: u13-aiter-agrs-full-capture-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
BRANCH=u13-aiter-agrs-full-capture-main
TESTS=(tests/distributed/test_comm_ops.py)
K='aiter'
PIP_EXTRA='ray'
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
