#!/bin/sh
# `rank` — mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# `|| rc=$?` and not `; rc=$?`: under `set -e` a simple command exiting
# non-zero kills the script before the assignment runs, so the branch below
# was never reached and a real-mode task died with the mock's status.
# A `||` puts it in a condition context, where `set -e` does not fire.
rc=0
bash "$PKG/assets/lib/mock.sh" stage3-analyze kernel_worklist || rc=$?
# 0 = mocked and written; 3 = this stage is not mocked, fall through to the
# real work; anything else is a mock that failed and must not be read as
# either.
if [ "$rc" -eq 0 ]; then
  # MOCK-MAP (A) + CONTRACT 3.4. `mock.sh` copies the sealed bytes faithfully
  # and on purpose; three things the sealed artefact could not have carried are
  # added here, as a step after the copy. m1 hit this, then m2, then me — one
  # lesson, three owners. Measured against the run at
  # runroot/runs/20260903T150709-4be7ad, where this kind was `invalid` on
  # check_environment AND, independently, on check_worklist_shape.
  exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/lib/m3_mock_adapt.py" \
    --kind kernel_worklist --out "${AGENT_SYS_OUTPUT_KERNEL_WORKLIST:?the runner exports this for an output slot}"
fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/rank.task/rank.py"
