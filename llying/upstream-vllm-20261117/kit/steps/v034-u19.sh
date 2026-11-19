#!/usr/bin/env bash
# v034-u19: unit tests of u19-rocm-pin-worker-slots on V034, against v0.34.0
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u19-rocm-pin-worker-slots rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/v0.34.0 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_V034 BRANCH=u19-rocm-pin-worker-slots BASE_WT=v0.34.0 BASE_REF=v0.34.0 NEED=v034
TESTS=(tests/v1/kv_offload/cpu/test_shared_offload_region.py tests/v1/kv_offload/tiering)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
