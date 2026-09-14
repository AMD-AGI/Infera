#!/bin/sh
# Assemble the deliverable. Reads nine handoffs and writes one directory in the
# experiment-result-packup layout.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

exec python3 "$PKG/assets/packup.task/packup.py" \
  --out "${AGENT_SYS_OUTPUT_INTEGRATION_PACKUP:?}" \
  --package "$PKG"
