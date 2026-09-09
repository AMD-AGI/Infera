#!/bin/sh
# The body of `check_packup_shape`. Counts substance rather than checking
# presence: the layout reference ships templates, so a packup with every mandated
# file and nothing in them is exactly what a template filled in by nobody looks
# like.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_packup_shape.validator/check.py"
