#!/bin/sh
# The body of `check_facts`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_facts.validator/entry.sh` with `cwd` set to a freshly
# allocated validation zone holding `args.json` and `inputs.json`.
#
# The environment is `ValidationEnvironment.env`, which is
# `{**config.values, TMPDIR, HOME, PWD}`. `PATH` is absent from that block and
# `sh` supplies a built-in default anyway — measured, so the fallback below is
# for determinism and not for reachability. What is genuinely missing is a route
# to the content: this body is handed handoff **ids** and nothing that resolves
# one. `AGENT_SYS_DEMO_STORE` arrives through `validation_env`, which
# `PhaseRunner._build_environment` already resolves. F-D5 in `demo/README.md`;
# a workaround, not a mechanism.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the demo runner exports one of these}}/assets/check_facts.validator/check.py"
