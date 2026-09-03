#!/usr/bin/env bash
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
bash "$PKG/assets/lib/mock.sh" stage5-integration stock.measurement:acceptance_stock patched.measurement:acceptance_patched integration_report && exit 0
echo "TODO(owner): integrate_and_verify has no real body yet; run with E2E_MOCK_STAGES covering stage5-integration" >&2
exit 1
