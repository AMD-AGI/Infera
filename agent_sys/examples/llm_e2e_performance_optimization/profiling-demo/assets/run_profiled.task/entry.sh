#!/bin/sh
# The profiled load: replay the trace and cut a torch-profiler window out of the
# middle of it.
#
# Two outputs from one task, because the profiler window has to fall inside the
# load window. As two sibling tasks agent_sys would schedule them concurrently
# with nothing to synchronise them, so lining them up would need a rendezvous
# file -- an edge the graph cannot see. See assets/load/replay.sh.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_LOAD_ROUND=profiled \
PD_CAPTURE=1 \
PD_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_AIPERF_PROFILED:?AGENT_SYS_OUTPUT_AIPERF_PROFILED is unset}" \
PD_OUTPUT_TRACE="${AGENT_SYS_OUTPUT_TORCH_TRACE:?AGENT_SYS_OUTPUT_TORCH_TRACE is unset}" \
exec /usr/bin/env bash "$PKG/assets/load/replay.sh"
