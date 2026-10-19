#!/bin/sh
# The execution entry point for `score`, the subgraph's `is_end` entry.
#
# `agent.backends.program.ProgramExecutor` runs this as `/bin/sh <entry>` with
# `cwd` set to the task's zone, so nothing here may assume a working directory.
#
# `AGENT_SYS_TASK_PACKAGE` is the staged copy of the package, inside the zone
# (`docs/interfaces.md` §4.16) — a task body reads the copy and needs nothing
# outside its own zone granted.
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:?}/assets/score.task/score.py"
