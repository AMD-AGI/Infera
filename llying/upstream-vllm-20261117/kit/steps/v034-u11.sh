#!/usr/bin/env bash
# v034-u11: unit tests of u11-offload-pp-worker-bytes on V034, against v0.34.0
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u11-offload-pp-worker-bytes rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/v0.34.0 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_V034 BRANCH=u11-offload-pp-worker-bytes BASE_WT=v0.34.0 BASE_REF=v0.34.0 NEED=v034
TESTS=(tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_kv_cache_planning.py tests/v1/kv_connector/unit/offloading_connector tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
