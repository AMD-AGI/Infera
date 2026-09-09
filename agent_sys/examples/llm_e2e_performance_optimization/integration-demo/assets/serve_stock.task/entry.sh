#!/bin/sh
# The stock arm: the mount plan is read and none of it is applied.
#
# Both serve tasks are the same script with one variable flipped, and that is the
# experimental design rather than a convenience -- a second implementation would
# be a second place for the two arms to differ.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

IT_ARM=stock \
IT_OUTPUT_DIR="${AGENT_SYS_OUTPUT_DEPLOYMENT_STOCK:?AGENT_SYS_OUTPUT_DEPLOYMENT_STOCK is unset}" \
exec /usr/bin/env bash "$PKG/assets/serve/round.sh"
