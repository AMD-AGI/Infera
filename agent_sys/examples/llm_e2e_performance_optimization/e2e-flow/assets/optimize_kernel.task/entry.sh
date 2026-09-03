#!/usr/bin/env bash
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
bash "$PKG/assets/lib/mock.sh" stage4-kernel-opt kernel_optimization && exit 0
echo "TODO(owner): optimize_kernel has no real body yet; run with E2E_MOCK_STAGES covering stage4-kernel-opt" >&2
exit 1
