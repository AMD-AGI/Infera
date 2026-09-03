#!/bin/sh
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
#
# `#!/bin/sh` + `set -eu`, per CONTRACT §3.2a — see the sibling body.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `kernel_table` is reshaped after it is copied: the sealed sample is a
# `reproducible` handoff and this kind is `structured_text`. MOCK-MAP (B).
rc=0
bash "$PKG/assets/lib/mock.sh" stage2-profiling \
  profiling_mode_on.bench_result:aiperf_profiled \
  profiling_mode_on.profile_result:torch_trace || rc=$?
if [ "$rc" = 0 ]; then
  python3 "$PKG/assets/lib/m2_reshape.py" kernel_table \
    "${E2E_MOCK_ROOT:?}/stage2-profiling/kernel_table/content" \
    "${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_KERNEL_TABLE:?}"
  # The two `reproducible` outputs carry a sealed `items/command` that does not
  # parse under its own shebang's shell; `mock.sh` repairs it for every mocked
  # kind (MOCK-MAP J), so nothing is needed here. `kernel_table` is
  # `structured_text` and carries no command item at all.
  #
  # **MOCK-MAP (A) — see the sibling body for why it belongs here.** All three
  # outputs need it and **the content type differs per kind**, which is the one
  # thing that cannot be a single loop over the slots: `env_render` writes to
  # `items/env/environment.yaml` for a `reproducible` kind and for a
  # `structured_text` one, but it is told which so that a kind whose type
  # changes fails loudly instead of silently writing to the old path.
  KIT_ENV="${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml"
  for spec in \
    "reproducible:${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT:?}" \
    "reproducible:${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_PROFILE_RESULT:?}" \
    "structured_text:${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_KERNEL_TABLE:?}"
  do
    python3 "$PKG/assets/lib/env_render.py" --inherit "$KIT_ENV" \
      --content-type "${spec%%:*}" --out "${spec#*:}"
  done
  exit 0
fi
if [ "$rc" != 3 ]; then exit "$rc"; fi

E2E_MODE=profiling_mode_on \
E2E_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT:?}" \
E2E_OUTPUT_TRACE="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_PROFILE_RESULT:?}" \
E2E_OUTPUT_KERNEL_TABLE="${AGENT_SYS_OUTPUT_PROFILING_MODE_ON_KERNEL_TABLE:?}" \
exec bash "$PKG/assets/load/line.sh"
