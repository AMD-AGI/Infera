#!/bin/sh
set -eu
exec python3 "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?}}/assets/downstream.task/process.py"
