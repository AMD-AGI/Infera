#!/bin/sh
# The profiler-detached line: CUDA graph ON, no profiler attached.
#
# **These are the numbers that mean something.** They are what stage 5's stock
# arm has to reproduce (M5.1.3.1), and they are the only throughput in this flow
# worth quoting — the profiler-attached line runs with graphs off and measured
# 8x slower on the sealed pair, which is the intent and not a regression.
#
# One task, not two: it brings its own service up and tears it down (M2.5).
# Everything else it shares byte for byte with the other line — see
# `../load/line.sh`, which is both of them.
#
# `#!/bin/sh` and `set -eu`, not bash and not `pipefail`: agent_sys never
# consults a body's shebang, it invokes one as `["/bin/sh", entry]`
# (`agent/backends/program.py:83`), and `/bin/sh` here is dash, which exits 2 on
# `set -o pipefail` before line 3 runs (CONTRACT §3.2a). `line.sh` is invoked
# with `bash` explicitly below, because it does use arrays.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `|| rc=$?` and not `&& exit 0`: `mock.sh` exits **3** when this stage is not
# in `$E2E_MOCK_STAGES`, and 0 only when it actually wrote something. The `||`
# is what keeps `set -e` from killing the script before the assignment runs.
rc=0
bash "$PKG/assets/lib/mock.sh" stage2-profiling \
  profiling_mode_off.bench_result:aiperf_baseline || rc=$?
if [ "$rc" = 0 ]; then exit 0; fi
if [ "$rc" != 3 ]; then exit "$rc"; fi

E2E_MODE=profiling_mode_off \
E2E_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_PROFILING_MODE_OFF_BENCH_RESULT:?}" \
exec bash "$PKG/assets/load/line.sh"
