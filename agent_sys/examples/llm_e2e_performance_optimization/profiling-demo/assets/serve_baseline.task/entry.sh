#!/bin/sh
# The baseline round: decode CUDA graphs ON, profiling control plane OFF.
#
# Both serve tasks are the same script with two flags flipped, so the round is
# named here and everything else lives in assets/serve/round.sh. Keeping the two
# rounds on one implementation is what stops them drifting into two deployments
# that differ in more than the axis under test.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_ROUND=baseline \
PD_CUDA_GRAPH=1 \
PD_PROFILE=0 \
PD_OUTPUT_DIR="${AGENT_SYS_OUTPUT_DEPLOYMENT_BASELINE:?AGENT_SYS_OUTPUT_DEPLOYMENT_BASELINE is unset}" \
exec /usr/bin/env bash "$PKG/assets/serve/round.sh"
