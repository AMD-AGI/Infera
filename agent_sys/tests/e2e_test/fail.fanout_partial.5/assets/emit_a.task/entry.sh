#!/bin/sh
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?}}/assets/emit_a.task/emit.py"
