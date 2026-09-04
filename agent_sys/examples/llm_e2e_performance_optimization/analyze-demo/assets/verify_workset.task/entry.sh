#!/bin/sh
# Its own wall-clock limit: `agent/backends/program.py` imposes none, and the
# CLI's settle bound is about the graph being quiet rather than about one task.
# Sized for five operators at the per-operator timeout, plus staging.
set -eu
BUDGET=$(( ${AD_PER_OP_TIMEOUT_S:-900} * 8 ))
exec timeout "$BUDGET" "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/verify_workset.task/verify.py"
