#!/bin/sh
# The body of `check_compiles`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_compiles.validator/entry.sh`, with `cwd` set to a
# freshly allocated validation zone holding `args.json`, `inputs.json` and
# `materials.json`.
#
# `PATH` is absent from `ValidationEnvironment.env`; `sh` supplies a built-in
# default, so the `python3` fallback below is for determinism rather than for
# reachability. `AGENT_SYS_TASK_PACKAGE` is what resolves this package's root
# from inside a zone that is nowhere near it.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_compiles.validator/check.py"
