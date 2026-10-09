#!/bin/sh
# The body of `check_directions`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_directions.validator/entry.sh`, with `cwd`
# set to a freshly allocated validation zone holding `args.json`, `inputs.json`
# and `materials.json`.
#
# **Both fallbacks are load-bearing, and which one fires depends on the phase.**
# `AGENT_SYS_TASK_PACKAGE` is a task body's variable. A validator has no prepared
# environment: `validator.choose_configuration` gives an INPUT phase the global
# row (`validator/environment.py:116-142`), and that row is
# `{PATH, AGENT_SYS_DEMO_PACKAGE, AGENT_SYS_DEMO_STORE, AGENT_SYS_DEMO_PYTHON}`
# (`cli/main.py:601-615`). So this validator resolves the second name in
# `problems`' input phase and would resolve neither if the line were shortened.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_directions.validator/check.py"
