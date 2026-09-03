#!/bin/sh
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# `|| rc=$?` and not `; rc=$?`: under `set -e` a simple command exiting
# non-zero kills the script before the assignment runs, so the branch below
# was never reached and a real-mode task died with the mock's status.
# A `||` puts it in a condition context, where `set -e` does not fire.
rc=0
bash "$PKG/assets/lib/mock.sh" stage3-analyze operator_identity || rc=$?
# 0 = mocked and written; 3 = this stage is not mocked, fall through to the
# real work; anything else is a mock that failed and must not be read as
# either.
if [ "$rc" -eq 0 ]; then exit 0; fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi
echo "TODO(owner): identify has no real body yet; run with E2E_MOCK_STAGES covering stage3-analyze" >&2
exit 1
