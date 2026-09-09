#!/bin/sh
# The body of `check_patch_shape`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_patch_shape.validator/entry.sh`, with `cwd`
# set to a freshly allocated validation zone holding `args.json`, `inputs.json`
# and `materials.json`.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_patch_shape.validator/check.py"
