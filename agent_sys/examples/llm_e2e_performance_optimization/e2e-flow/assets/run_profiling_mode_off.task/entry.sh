#!/usr/bin/env bash
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
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

bash "$PKG/assets/lib/mock.sh" stage2-profiling \
  profiling_mode_off.bench_result:aiperf_baseline && exit 0

E2E_MODE=profiling_mode_off \
E2E_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_PROFILING_MODE_OFF_BENCH_RESULT:?}" \
exec bash "$PKG/assets/load/line.sh"
