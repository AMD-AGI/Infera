#!/bin/sh
# The body of `check_deploy_reproduces`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_deploy_reproduces.validator/entry.sh`, with
# `cwd` set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# **Both environment rows, in one line.** `AGENT_SYS_TASK_PACKAGE` is what the
# PRODUCER row exports and `AGENT_SYS_DEMO_PACKAGE` is what the GLOBAL row
# exports; they are disjoint, not nested (`validator` spec §8.2), so a body
# naming only one of them dies on the other phase. This package reaches the
# producer row today -- `deploy_kit` has no consumer until stage 2 lands -- and
# the fallback costs nothing and removes a trap for whoever adds one.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_deploy_reproduces.validator/check.py"
