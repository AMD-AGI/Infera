#!/usr/bin/env bash
# nightly-u26: unit tests of u26-rocm-exact-pin-main on NIGHTLY, against nightly
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u26-rocm-exact-pin-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/nightly rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_NIGHTLY BRANCH=u26-rocm-exact-pin-main BASE_WT=nightly BASE_REF=9c1d45c21836370104d4eb51d3b46eda20719c47 NEED=nightly
TESTS=(tests/v1/kv_offload/cpu)
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
