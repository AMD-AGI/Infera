#!/bin/sh
# The exact execution entry point for `produce`.
#
# `agent.backends.program.ProgramExecutor` runs this as `/bin/sh <entry>` with
# `cwd` set to the task's zone, so nothing here may assume a working directory.
#
# **`AGENT_SYS_TASK_PACKAGE` first.** `interfaces.md` §4.16 stages the package
# into `<zone>/package/` and `env_mgr` exports the copy under that name in
# `Prepared.environment` — so a task body reads the copy, inside its own zone,
# and needs nothing outside it granted. `AGENT_SYS_DEMO_PACKAGE` is the fallback
# for a body run **without** a prepared environment: a validator body, which
# `ScriptBodyRunner` runs with `validation_env` instead.
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the demo runner exports one of these}}/assets/produce.task/collect.py"
