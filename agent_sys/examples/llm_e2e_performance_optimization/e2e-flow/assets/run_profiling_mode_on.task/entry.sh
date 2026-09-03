#!/usr/bin/env bash
# The profiler-attached line: CUDA graph OFF, profiling control plane ON.
#
# **CUDA graph off is not a mistake and not a second variable.** With graphs on
# the profiler records one graph launch instead of the kernels inside it, so a
# graphs-on capture cannot attribute time to a kernel — which is the only thing
# this line exists to produce. The cost is that its throughput is not a control
# for anything; the profiler-detached line is where that number comes from.
#
# Three outputs from one task, and they are not separable. The profiler window
# has to fall *inside* the load window, and the ranking reads the trace this
# task just wrote. As sibling tasks agent_sys would schedule them concurrently
# with nothing to synchronise them, so lining them up would need a rendezvous
# file — an edge the graph cannot see and cannot report on.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `kernel_table` is reshaped after it is copied: the sealed sample is a
# `reproducible` handoff and this kind is `structured_text`. See
# `../lib/m2_reshape.py`, and MOCK-MAP.md, whose row for this kind records only
# adaptation (A).
if bash "$PKG/assets/lib/mock.sh" stage2-profiling \
     profiling_mode_on.bench_result:aiperf_profiled \
     profiling_mode_on.profile_result:torch_trace; then
  KT="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_KERNEL_TABLE:?}"
  python3 "$PKG/assets/lib/m2_reshape.py" kernel_table \
    "${E2E_MOCK_ROOT:?}/stage2-profiling/kernel_table/content" "$KT"
  exit 0
fi

E2E_MODE=profiling_mode_on \
E2E_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT:?}" \
E2E_OUTPUT_TRACE="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_PROFILE_RESULT:?}" \
E2E_OUTPUT_KERNEL_TABLE="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_KERNEL_TABLE:?}" \
exec bash "$PKG/assets/load/line.sh"
