#!/bin/sh
# Run by `validator.ScriptBodyRunner` as `/bin/sh <package>/assets/<name>.validator/entry.sh`
# with `cwd` set to a fresh validation zone holding `args.json`, `inputs.json`
# and `materials.json`.
#
# The environment is `ValidationEnvironment.env` — `{**config.values, TMPDIR,
# HOME, PWD}` — so `PATH` is absent and the interpreter is named explicitly.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_kernel_table.validator/check.py"
