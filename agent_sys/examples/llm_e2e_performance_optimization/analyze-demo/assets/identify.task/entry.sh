#!/bin/sh
# `identify` may clone and index repositories, so it carries its own wall-clock
# limit: `agent/backends/program.py` polls the subprocess and imposes none, and
# the CLI's `_settle` bound is about the whole graph being quiet rather than
# about one task.
set -eu
exec timeout "${AD_RESOLVE_TIMEOUT_S:-1800}" "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/identify.task/identify.py"
