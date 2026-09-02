#!/bin/sh
# The body of `check_analysis`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_analysis.validator/entry.sh`, with `cwd` set to a
# freshly allocated validation zone holding `args.json`, `inputs.json` and
# `materials.json`.
#
# Same shape as `check_compiles.validator/entry.sh` and for the same reasons:
# `PATH` is absent from `ValidationEnvironment.env`, and `AGENT_SYS_TASK_PACKAGE`
# is what resolves this package's root from inside the zone.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_analysis.validator/check.py"
