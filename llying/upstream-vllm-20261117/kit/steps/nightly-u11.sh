#!/usr/bin/env bash
# nightly-u11: unit tests of u11-offload-pp-worker-bytes-main on NIGHTLY, against nightly
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u11-offload-pp-worker-bytes-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/nightly rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_NIGHTLY BRANCH=u11-offload-pp-worker-bytes-main BASE_WT=nightly BASE_REF=9c1d45c21836370104d4eb51d3b46eda20719c47 NEED=nightly
TESTS=(tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_kv_cache_planning.py tests/v1/kv_connector/unit/offloading_connector tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
