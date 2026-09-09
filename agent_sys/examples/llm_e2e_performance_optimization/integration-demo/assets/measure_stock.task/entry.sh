#!/bin/sh
# Measure the unpatched arm. Two outputs from one task, because correctness and
# performance have to run against the same deployment instance and must not
# overlap in time; as sibling tasks agent_sys would schedule them concurrently.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

IT_ARM=stock \
IT_OUTPUT_ACCEPT="${AGENT_SYS_OUTPUT_ACCEPTANCE_STOCK:?AGENT_SYS_OUTPUT_ACCEPTANCE_STOCK is unset}" \
IT_OUTPUT_BENCH="${AGENT_SYS_OUTPUT_BENCH_STOCK:?AGENT_SYS_OUTPUT_BENCH_STOCK is unset}" \
exec /usr/bin/env bash "$PKG/assets/accept/measure.sh"
