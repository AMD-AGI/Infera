#!/bin/sh
# The body of `check_solvable`. Run by `validator.ScriptBodyRunner` in a freshly
# allocated validation zone holding `args.json`, `inputs.json` and
# `materials.json`.
#
# This one reads nothing outside the artefact it is handed — no catalogue, no
# second handoff — so the package root is needed only to find this file's own
# sibling `check.py`. The fallback is kept identical to the other two rather than
# shortened: the same line in three places is one thing to check, and a validator
# that resolves its own script differently from its neighbours is a puzzle for
# whoever debugs the run.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_solvable.validator/check.py"
