#!/usr/bin/env bash
# v034-u26b: (rerun after the ROCm-only gating, 11-19 12:00) unit tests of u26-rocm-exact-pin on V034, against v0.34.0
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u26-rocm-exact-pin rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/v0.34.0 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_V034 BRANCH=u26-rocm-exact-pin BASE_WT=v0.34.0 BASE_REF=v0.34.0 NEED=v034
TESTS=(tests/v1/kv_offload/cpu)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
