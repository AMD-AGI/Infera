#!/usr/bin/env bash
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
bash "$PKG/assets/lib/mock.sh" stage1-deploy deploy_kit && exit 0
echo "TODO(owner): deploy_and_prove has no real body yet; run with E2E_MOCK_STAGES covering stage1-deploy" >&2
exit 1
