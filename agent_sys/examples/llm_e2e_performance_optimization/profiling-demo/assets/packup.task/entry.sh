#!/bin/sh
# Assemble the five upstream handoffs into one directory a colleague can follow.
#
# The only task that touches no cluster at all. Everything it needs was published
# by the tasks before it, which is the property that makes the packup a
# deliverable rather than a second copy of the run.
#
# The script is `assemble.sh` and not `packup.sh` on purpose: `spec_loader` finds
# a body by matching the closure's name against filenames under assets/, so a
# shared script named after its task is a second candidate for that task's entry
# and the load fails rather than guessing between them.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

PD_INPUT_DEPLOYMENT_BASELINE="${AGENT_SYS_INPUT_DEPLOYMENT_BASELINE:?}" \
PD_INPUT_DEPLOYMENT_PROFILED="${AGENT_SYS_INPUT_DEPLOYMENT_PROFILED:?}" \
PD_INPUT_AIPERF_BASELINE="${AGENT_SYS_INPUT_AIPERF_BASELINE:?}" \
PD_INPUT_AIPERF_PROFILED="${AGENT_SYS_INPUT_AIPERF_PROFILED:?}" \
PD_INPUT_KERNEL_TABLE="${AGENT_SYS_INPUT_KERNEL_TABLE:?}" \
PD_OUTPUT_PACKUP="${AGENT_SYS_OUTPUT_PROFILE_PACKUP:?}" \
exec /usr/bin/env bash "$PKG/assets/kit/assemble.sh"
