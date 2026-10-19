#!/bin/sh
# A validator's body. It runs with a fresh validation zone as the working
# directory; AGENT_SYS_DEMO_PYTHON is the interpreter the run itself uses.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-$AGENT_SYS_DEMO_PACKAGE}/assets/check_env_report.validator/check.py"
