#!/bin/sh
# The execution entry point for `seed_table`.
#
# `agent.backends.program.ProgramExecutor` runs this as `/bin/sh <entry>` with
# `cwd` set to the task's zone, so nothing here may assume a working directory.
#
# `AGENT_SYS_TASK_PACKAGE` names the staged copy of this package inside the
# zone; `AGENT_SYS_DEMO_PACKAGE` is the CLI's fallback for a body run without a
# prepared environment.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/seed_table.task/seed.py"
