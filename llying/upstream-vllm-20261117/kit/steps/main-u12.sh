#!/usr/bin/env bash
# main-u12: u12-gfx950-gluon-mqa-logits-main vs vLLM main (wt/main), nightly image with the trees' Python code synced in
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u12-gfx950-gluon-mqa-logits-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
BRANCH=u12-gfx950-gluon-mqa-logits-main
TESTS=(tests/kernels/attention/test_rocm_gluon_paged_mqa_logits.py tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py tests/kernels/attention/test_rocm_aiter_candidate_mask.py)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_main.sh
