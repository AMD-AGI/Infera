#!/bin/sh
# The body of `check_problems`. Run by `validator.ScriptBodyRunner` in a freshly
# allocated validation zone holding `args.json`, `inputs.json` and
# `materials.json`.
#
# The two-name fallback is the same one `check_directions/entry.sh` explains at
# length: a validator body has no prepared environment, so an INPUT phase sees
# only the global row, which carries `AGENT_SYS_DEMO_PACKAGE` and not
# `AGENT_SYS_TASK_PACKAGE`. Kept identical across all three bodies here.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_problems.validator/check.py"
