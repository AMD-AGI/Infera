#!/usr/bin/env bash
# v034-u12: unit tests of u12-gfx950-gluon-mqa-logits on V034, against v0.34.0
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u12-gfx950-gluon-mqa-logits rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/v0.34.0 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_V034 BRANCH=u12-gfx950-gluon-mqa-logits BASE_WT=v0.34.0 BASE_REF=v0.34.0 NEED=v034
TESTS=(tests/kernels/attention/test_rocm_gluon_paged_mqa_logits.py tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py tests/kernels/attention/test_rocm_aiter_candidate_mask.py)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
