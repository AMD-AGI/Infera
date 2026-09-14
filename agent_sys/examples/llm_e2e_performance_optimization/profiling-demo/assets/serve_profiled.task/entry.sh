#!/bin/sh
# The profiled round: decode CUDA graphs OFF, profiling control plane ON.
#
# Same script as serve_baseline with two flags flipped. It tears the baseline
# deployment down first — mix_up.sh always starts with an idempotent teardown and
# a VRAM gate — which is why this task depends on run_baseline rather than on
# serve_baseline: the edge says the baseline has been measured and is safe to
# replace.
#
# PD_REUSE_DEPLOYMENT is deliberately not honoured here even when it is set for
# the rest of the run. The baseline deployment is up and healthy at this point,
# so reusing it would skip the bring-up that is the entire content of this task
# and then hand the trace round a graphs-ON engine.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_ROUND=profiled \
PD_CUDA_GRAPH=0 \
PD_PROFILE=1 \
PD_REUSE_DEPLOYMENT=0 \
PD_OUTPUT_DIR="${AGENT_SYS_OUTPUT_DEPLOYMENT_PROFILED:?AGENT_SYS_OUTPUT_DEPLOYMENT_PROFILED is unset}" \
exec /usr/bin/env bash "$PKG/assets/serve/round.sh"
