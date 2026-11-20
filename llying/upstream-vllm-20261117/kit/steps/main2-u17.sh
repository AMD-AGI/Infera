#!/usr/bin/env bash
# main2-u17: u17-aiter-fused-shared-ep-main (on main 8828834fb) vs that main (wt/main2), nightly image with the trees'
# Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u17-aiter-fused-shared-ep-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main2.sh
BRANCH=u17-aiter-fused-shared-ep-main
TESTS=(tests/kernels/moe/test_rocm_aiter_topk.py tests/model_executor/layers/test_fused_shared_expert.py tests/kernels/moe/test_naive_dp_ep_dbo.py tests/quantization/test_online_shared_expert_fusion.py)
PER_FILE=1
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main2.sh
