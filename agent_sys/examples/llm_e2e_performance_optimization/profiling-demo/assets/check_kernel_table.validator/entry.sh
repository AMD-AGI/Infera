#!/bin/sh
# The body of `check_kernel_table`. Recomputes the head's share from the CSV
# rather than trusting the summary the producer wrote beside it.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_kernel_table.validator/check.py"
