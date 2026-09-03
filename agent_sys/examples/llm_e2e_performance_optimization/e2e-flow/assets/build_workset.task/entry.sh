#!/usr/bin/env bash
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
bash "$PKG/assets/lib/mock.sh" stage3-analyze operator_workset && exit 0
echo "TODO(owner): build_workset has no real body yet; run with E2E_MOCK_STAGES covering stage3-analyze" >&2
exit 1
