#!/bin/sh
# The body of `check_deploy_serves`. Run by `validator.ScriptBodyRunner` as
# `["/bin/sh", entry]` (`agent_sys/validator/phase.py:147`) — the shebang is
# never consulted, so this must be POSIX. `/bin/sh` is dash here and
# `set -o pipefail` is a hard exit 2 on line 1 under it, which arrives as an
# UNREACHED phase rather than as a verdict.
#
# **Both environment rows, in one expression.** `AGENT_SYS_TASK_PACKAGE` is what
# the PRODUCER row exports and `AGENT_SYS_DEMO_PACKAGE` is what the GLOBAL row
# exports; they are disjoint, not nested (`validator` spec §8.2), so a body
# naming only one of them dies on the other phase. This has already cost a run.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_deploy_serves.validator/check.py"
