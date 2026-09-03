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
if [ "$rc" -eq 0 ]; then
  # MOCK-MAP (A) + CONTRACT 3.4. `mock.sh` copies the sealed bytes faithfully
  # and on purpose; three things the sealed artefact could not have carried are
  # added here, as a step after the copy. m1 hit this, then m2, then me — one
  # lesson, three owners. Measured against the run at
  # runroot/runs/20260903T150709-4be7ad, where this kind was `invalid` on
  # check_environment AND, independently, on check_worklist_shape.
  exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/lib/m3_mock_adapt.py" \
    --kind operator_identity --out "${AGENT_SYS_OUTPUT_OPERATOR_IDENTITY:?the runner exports this for an output slot}"
fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi
exec timeout "${E2E_RESOLVE_TIMEOUT_S:-1800}" "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "$PKG/assets/identify.task/identify.py"
