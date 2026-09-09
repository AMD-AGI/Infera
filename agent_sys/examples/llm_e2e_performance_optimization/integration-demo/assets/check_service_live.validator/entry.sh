#!/bin/sh
# The body of `check_service_live`. Run by `validator.ScriptBodyRunner` as
# `/bin/sh <package_root>/assets/check_service_live.validator/entry.sh`, with
# `cwd` set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# It reads only files: the deployment record it is handed. It does not call the
# endpoint. A validator that re-probed a live service would pass or fail on the
# state of the cluster at validation time rather than on the artefact, and the
# same handoff would then get different verdicts on different days.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_service_live.validator/check.py"
