#!/bin/sh
# The body of `check_scores`. `validator.ScriptBodyRunner` runs it as
# `/bin/sh <package_root>/assets/check_scores.validator/entry.sh` with `cwd`
# set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# **Two variables, because two rows of the configuration chain can reach here.**
# `AGENT_SYS_TASK_PACKAGE` is the staged copy and arrives when the phase takes
# §8.2's PRODUCER row — the producing task's resolved environment.
# `AGENT_SYS_DEMO_PACKAGE` is the checkout and arrives on the GLOBAL row, which
# is what an *input* phase takes: `_configuration_sources` supplies no
# `consumer`, so `choose_configuration` falls through
# (`validator/phase.py:297-358`). `AGENT_SYS_DEMO_PYTHON` is on the global row
# too; `python3` is the fallback for the other.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_scores.validator/check.py"
