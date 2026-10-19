#!/bin/sh
# A program task's body. It runs as `/bin/sh entry.sh` with the task's zone as
# the working directory; AGENT_SYS_TASK_PACKAGE is the package copy staged there.
set -eu
exec python3 "$AGENT_SYS_TASK_PACKAGE/assets/env_facts.task/collect.py"
