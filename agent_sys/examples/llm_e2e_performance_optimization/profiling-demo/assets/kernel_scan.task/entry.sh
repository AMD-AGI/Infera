#!/bin/sh
# Rank the kernels in the captured trace.
#
# The only task in this package whose work is reading rather than running: it
# takes the trace handoff and produces the table the operator-selection stage
# consumes. It still reaches the cluster, because Magpie's dependencies are
# installed on the compute node and not on the login node.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_INPUT_TORCH_TRACE="${AGENT_SYS_INPUT_TORCH_TRACE:?AGENT_SYS_INPUT_TORCH_TRACE is unset}" \
PD_OUTPUT_KERNEL_TABLE="${AGENT_SYS_OUTPUT_KERNEL_TABLE:?AGENT_SYS_OUTPUT_KERNEL_TABLE is unset}" \
exec /usr/bin/env bash "$PKG/assets/analyze/scan.sh"
