#!/bin/sh
# The body of `check_packup_shape`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_packup_shape.validator/entry.sh`, with
# `cwd` set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# **Both environment rows, in one line.** `AGENT_SYS_TASK_PACKAGE` is what the
# PRODUCER row exports and `AGENT_SYS_DEMO_PACKAGE` is what the GLOBAL row
# exports; they are disjoint, not nested
# (`scratch/single-real-task-2026-08/validator-env.md` §5), so a body naming
# only one of them dies on the other phase. This package only ever reaches the
# producer row today — `runbook` has no consumer — and the fallback costs
# nothing and removes a trap for whoever adds one.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_packup_shape.validator/check.py"
