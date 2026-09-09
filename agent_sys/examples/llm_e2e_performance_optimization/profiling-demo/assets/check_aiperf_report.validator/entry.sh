#!/bin/sh
# The body of `check_aiperf_report`. Run by `validator.ScriptBodyRunner` with cwd
# set to a validation zone holding args.json, inputs.json and materials.json.
# Reads files only; it does not send a request of its own.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_aiperf_report.validator/check.py"
