#!/bin/sh
# The exact execution entry point for `consume`. Never reached in this demo.
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the demo runner exports one of these}}/assets/consume.task/render.py"
