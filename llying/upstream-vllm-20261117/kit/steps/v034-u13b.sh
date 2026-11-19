#!/usr/bin/env bash
# v034-u13b: (rerun with ray installed; v034-u13 collected nothing) unit tests of u13-aiter-agrs-full-capture on V034, against v0.34.0
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/v0.34.0 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
IMG=$IMG_V034 BRANCH=u13-aiter-agrs-full-capture BASE_WT=v0.34.0 BASE_REF=v0.34.0 NEED=v034
TESTS=(tests/distributed/test_comm_ops.py)
K='aiter'
PIP_EXTRA=ray
. /home/liyingli/bench_agentx/upstream-20261119/steps/_vllm_unit.sh
