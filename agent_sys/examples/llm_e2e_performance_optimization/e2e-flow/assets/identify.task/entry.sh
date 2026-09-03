#!/bin/sh
# `identify` — mock first, real work second. It may clone and index
# repositories, so it carries its own wall-clock limit:
# `agent/backends/program.py` polls the subprocess and imposes none, and the
# CLI's `_settle` bound is about the whole graph being quiet rather than about
# one task.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
rc=0
bash "$PKG/assets/lib/mock.sh" stage3-analyze operator_identity || rc=$?
if [ "$rc" -eq 0 ]; then exit 0; fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi
exec timeout "${E2E_RESOLVE_TIMEOUT_S:-1800}" "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "$PKG/assets/identify.task/identify.py"
