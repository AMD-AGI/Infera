#!/bin/sh
# The baseline load: replay the trace against the graphs-on deployment, with no
# profiler attached. This is the round whose throughput is worth quoting.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_LOAD_ROUND=baseline \
PD_CAPTURE=0 \
PD_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_AIPERF_BASELINE:?AGENT_SYS_OUTPUT_AIPERF_BASELINE is unset}" \
exec /usr/bin/env bash "$PKG/assets/load/replay.sh"
