#!/usr/bin/env bash
# nightly-u13: unit tests of u13-aiter-agrs-full-capture-main on NIGHTLY, against nightly
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/nightly rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_NIGHTLY BRANCH=u13-aiter-agrs-full-capture-main BASE_WT=nightly BASE_REF=9c1d45c21836370104d4eb51d3b46eda20719c47 NEED=nightly
TESTS=(tests/distributed/test_comm_ops.py)
K='aiter'
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
