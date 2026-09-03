#!/usr/bin/env bash
# SKELETON. Mock first, real work second. See ../../CONTRACT.md and ../../MOCK-MAP.md.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
bash "$PKG/assets/lib/mock.sh" stage2-profiling profiling_mode_on.bench_result:aiperf_profiled profiling_mode_on.profile_result:torch_trace profiling_mode_on.kernel_table:kernel_table && exit 0
echo "TODO(owner): run_profiling_mode_on has no real body yet; run with E2E_MOCK_STAGES covering stage2-profiling" >&2
exit 1
