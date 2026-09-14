#!/bin/sh
# The body of `check_reproduces`. Same shape as `check_packup_shape`'s and for
# the same reasons: `AGENT_SYS_TASK_PACKAGE` is the PRODUCER row's package
# variable and `AGENT_SYS_DEMO_PACKAGE` is the GLOBAL row's, and the two rows
# are disjoint rather than nested.
#
# **Nothing here reads a credential.** `ANTHROPIC_API_KEY` and
# `ANTHROPIC_BASE_URL` arrive in this body's environment by name, through the
# operator's own `~/.claude/settings.json` allow-list, and are consumed by
# `claude` itself. No value is read into a variable, echoed, or written to a
# file by this package.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_reproduces.validator/check.py"
