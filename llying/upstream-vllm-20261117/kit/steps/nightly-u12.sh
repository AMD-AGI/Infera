#!/usr/bin/env bash
# nightly-u12: unit tests of u12-gfx950-gluon-mqa-logits-main on NIGHTLY, against nightly
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u12-gfx950-gluon-mqa-logits-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/nightly rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_NIGHTLY BRANCH=u12-gfx950-gluon-mqa-logits-main BASE_WT=nightly BASE_REF=9c1d45c21836370104d4eb51d3b46eda20719c47 NEED=nightly
TESTS=(tests/kernels/attention/test_rocm_gluon_paged_mqa_logits.py tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py tests/kernels/attention/test_rocm_aiter_candidate_mask.py)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
